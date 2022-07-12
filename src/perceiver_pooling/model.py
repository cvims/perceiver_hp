#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# =============================================================================
"""
Contains Multi-Modal Multi-Instance Perceiver Hopfield/Mean Pooling architecture.
"""
# =============================================================================
# Imports
# =============================================================================
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from cvims.data.loaders import BeautifyDataLoaderIterations, create_data_loader
from cvims.network.operator import NetworkOperator
from src.parse_config import parse_arguments
from cvims.models.perceiver import MultiInputPerceiver
from cvims.network import EarlyStoppingByLoss, get_model_trainable_parameter_count
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.nn import Sequential, Linear, LayerNorm, ModuleDict, ModuleList, Dropout
from torch.utils.tensorboard import SummaryWriter
from einops.layers.torch import Reduce, Rearrange
from hflayers import HopfieldPooling
from typing import Dict, List, Any, Tuple, Iterable


class PerceiverPooling(torch.nn.Module):
    """
    Architecture for multiple perceivers with hopfield/mean pooling for mutli modal and multi instance processing
    """
    def __init__(self, perceivers_parameters: Any, general_model_parameters: Any, output_classes: int = 10) -> None:
        """

        :param perceivers_parameters: Information about every single perceiver strcture for multi modality
        :param general_model_parameters: Information for the overall model, e. g. fusion information.
        """
        assert perceivers_parameters and general_model_parameters
        super().__init__()

        # hopfield pooling or mean pooling?
        self.use_hopfield_pooling_fusion = general_model_parameters['use_hopfield_pooling_fusion']

        # fusion depth for hopfield / mean pooling
        self.fusion_n = general_model_parameters['fusion_n']

        # perceiver architecture
        self.perceivers = ModuleDict({})
        self.perceivers.forward = self._perceivers_forward
        self.post_perceivers = ModuleDict({})
        self.post_perceivers.forward = self._post_perceivers_forward        

        # store dropouts for forward pass
        self.perceiver_dropouts = {}

        # build the structure of every single perceiver
        for perceiver_name in perceivers_parameters:
            # are dropouts set?
            dropout_name = '_'.join([perceiver_name, 'dropout'])
            if dropout_name in general_model_parameters:
                self.perceiver_dropouts[perceiver_name] = float(general_model_parameters[dropout_name])

            # get model parameters for the perceiver
            model_params = perceivers_parameters[perceiver_name]

            use_hopfield_pooling = model_params['use_hopfield_pooling']

            # create a multi input perceiver (hopfield pooling is applied after that)
            self.perceivers[perceiver_name] = MultiInputPerceiver(
                **{k: model_params[k] for k in model_params if 'hopfield' not in str(k)}
            )

            # reduce the latent index dimension of the perceiver output by taking the mean as in the traditional transformer architecture
            post_perceiver_modules = ModuleList([])
            post_perceiver_modules.append(LayerNorm(model_params['latent_dim']))
            post_perceiver_modules.append(Reduce(pattern='b i n d -> b i d', reduction='mean'))

            # hopfield or mean pooling of instances per perceiver (this is not the modality fusion!)
            if use_hopfield_pooling:
                # use hopfield pooling for instances
                post_perceiver_modules.append(
                    HopfieldPooling(
                        input_size=model_params['latent_dim'],
                        output_size=self.fusion_n,
                        hidden_size=model_params['hopfield_dim_head'],
                        num_heads=model_params['hopfield_heads'],
                        dropout=model_params['hopfield_dropout'],
                        scaling=model_params['hopfield_scaling'],
                        update_steps_max=model_params['hopfield_max_update_steps'],
                        )
                )
                post_perceiver_modules.append(
                    LayerNorm(self.fusion_n)
                )
            else:
                # use mean pooling for instances
                post_perceiver_modules.append(
                    Reduce(pattern='b i d -> b d', reduction='mean'),
                )
                post_perceiver_modules.append(
                    LayerNorm(model_params['latent_dim'])
                )
                # to be able the fuse them later, we need the same output dim for each individual perceiver
                post_perceiver_modules.append(
                    Linear(
                        in_features=model_params['latent_dim'],
                        out_features=self.fusion_n)
                )
                post_perceiver_modules.append(Dropout(p=0.2))

            # Hopfield pooling and mean pooling do not have an instance dimension anymore. We need them again for the
            # fusion mechanism. Now we rearrange the post perceiver output to get an instance dimension of size 1.
            post_perceiver_modules.append(Rearrange('(b i) f -> b i f', i=1))

            # add to post perceiver module dict
            self.post_perceivers[perceiver_name] = Sequential(*post_perceiver_modules)

        # We also distinguish between hopfield pooling and mean pooling for fusion
        # You can also set the config in a way that you use mean pooling and hopfield pooling more flexible, e. g. to
        # use mean pooling only for modality fusion but not for instance fusion, etc.
        if self.use_hopfield_pooling_fusion:
            # we pool over the instances and keep the output size equal to the input size
            self.fusion_layer = HopfieldPooling(
                input_size=self.fusion_n,
                output_size=self.fusion_n,
                hidden_size=general_model_parameters['hopfield_fusion_dim_head'],
                num_heads=general_model_parameters['hopfield_fusion_heads'],
                dropout=general_model_parameters['hopfield_fusion_dropout'],
                scaling=general_model_parameters['hopfield_fusion_scaling'],
                update_steps_max=general_model_parameters['hopfield_fusion_max_update_steps']
            )
        else:
            # we mean over the instance representations
            self.fusion_layer = Reduce('b i n -> b n', reduction='mean')

        # projection from fusion layer to output layer
        self.output_layer = Sequential(
            LayerNorm(self.fusion_n),
            Linear(self.fusion_n, output_classes)
        )

        self.perceivers_parameters = perceivers_parameters
        self.general_model_parameters = general_model_parameters

    def copy_weights_to_individual_modality_network(self, empty_model, modality: str):
        """
        :param empty_model: Initialized individual model with unassigned weights
        :param modality
        """
        perceiver_weights = self.perceivers[modality].state_dict()

        empty_model.perceivers[modality].load_state_dict(perceiver_weights)

        if self.post_perceivers is not None:
            if modality in self.post_perceivers:
                post_perceiver_weights = self.post_perceivers[modality].state_dict()
                empty_model.post_perceivers[modality].load_state_dict(post_perceiver_weights)

        if self.fusion_layer is not None:
            fusion_layer_weights = self.fusion_layer.state_dict()
            empty_model.fusion_layer.load_state_dict(fusion_layer_weights)

        if self.output_layer is not None:
            output_layer_weights = self.output_layer.state_dict()
            empty_model.output_layer.load_state_dict(output_layer_weights)


    def create_individual_perceiver_model(self, perceiver_key: str or List[str], keep_weights: bool):
        """
        Creates a mew model with one or more perceiver keys
        :param perceiver_key: Modality name
        :param keep_weights: Boolean flag to keep weights (True) or keep it untrained (False)
        :return: PerceiverPooling object
        """
        assert isinstance(perceiver_key, str) or isinstance(perceiver_key, list)

        if isinstance(perceiver_key, str):
            perceivers_parameters = {perceiver_key: self.perceivers_parameters[perceiver_key]}
        else:
            perceivers_parameters = {k: self.perceivers_parameters[k] for k in perceiver_key}

        p_fusion = PerceiverPooling(
            perceivers_parameters=perceivers_parameters,
            general_model_parameters=self.general_model_parameters
        )

        if keep_weights:
            for p_name in perceivers_parameters:
                self.copy_weights_to_individual_modality_network(
                    empty_model=p_fusion, modality=p_name
                )

        p_fusion.to(device=next(self.parameters()).device)

        return p_fusion

    def _perceivers_forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of perceivers
        :param data: Dictionary with modality name as key and tensor for instances as value
        :return:
        """
        # shuffle the keys to apply dropout (if activated) for multiple modalities randomly
        _keys = list(data.keys())
        r_ids = torch.randperm(n=len(list(data.keys())))
        data_keys = []
        for r_id in r_ids:
            data_keys.append(_keys[r_id])

        # return variable
        perceiver_outputs = {}

        # control the modality dropout config setting
        perceiver_dropout_available = True

        for i, data_name in enumerate(data_keys):
            if data_name not in self.perceivers:
                continue

            # we want to keep at least one modality in case all
            if not (perceiver_dropout_available and i == len(data) - 1):
                # dropout available
                if data_name in self.perceiver_dropouts:
                    dropout_value = self.perceiver_dropouts[data_name]
                    if torch.rand(1) < dropout_value:
                        # skip perceiver
                        continue

            # apply data from modality to the corresponding perceiver
            d = data[data_name]
            perceiver = self.perceivers[data_name]
            x = perceiver(d)
            perceiver_outputs[data_name] = torch.clone(x)
            # at least one perceiver got traversed
            perceiver_dropout_available = False

        return perceiver_outputs

    def _post_perceivers_forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Applies post perceiver activity (hopfield pooling or mean pooling)
        :param data: Dictionary with modality name as key and tensor for instances as value
        :return:
        """
        post_perceivers_outputs = ()
        for data_name in data:
            perceiver_output = data[data_name]
            post_perceiver_layer = self.post_perceivers[data_name]
            x = post_perceiver_layer(perceiver_output)
            post_perceivers_outputs += (x,)

        # cat them so that the instance dimension is behind the batch size dim again
        x = torch.cat(post_perceivers_outputs, dim=1)

        return x

    def get_attention_weights(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor] or None:
        """
        Access the attention weights per modality of hopfield pooling.
        """
        def associate_attention_fusion_weights(perceiver_names, fusion_weights: Dict):
            attention_fusion_weights = {p: [] for p in perceiver_names}
            for perceiver_name in perceiver_names:
                if perceiver_name in fusion_weights:
                    attention_fusion_weights[perceiver_name].append(fusion_weights[perceiver_name])
                else:
                    # append 0.0 to make see which of the perceivers had a dropout at a specific time
                    attention_fusion_weights[perceiver_name].append(torch.tensor(0.0, dtype=torch.float32))
                attention_fusion_weights[perceiver_name] = torch.stack(attention_fusion_weights[perceiver_name])

            return attention_fusion_weights

        if not self.use_hopfield_pooling_fusion:
            # Cannot get attention weights without having an hopfield pooling layer
            return None
        else:
            # traverse perceivers and post perceivers
            x_dict = self.perceivers(data)

            x = self.post_perceivers(x_dict)

            data_order = list(x_dict.keys())
            # bsz, heads, tgt len, src len
            fusion_weights = self.fusion_layer.get_association_matrix(x)
            # mean head dim then mean over batch dim
            fusion_weights = fusion_weights.mean(dim=1).mean(dim=0).squeeze(dim=0)

            return associate_attention_fusion_weights(
                perceiver_names=list(self.perceivers.keys()),
                fusion_weights={a: b for a, b in zip(data_order, fusion_weights)}
            )

    def forward(self, x) -> torch.Tensor:
        # returns dict of output of each modality
        x = self.perceivers(x)

        # returns dict of output of every post processing of each perceiver modality
        x = self.post_perceivers(x)

        # modality fusion output
        x = self.fusion_layer(x)

        # classification output
        x = self.output_layer(x)

        return x


def run_network(parsed_config: Any):
    """
    Uses config file to load modality perceiver configuration as well as fusion information (see ./configs/*)
    :config: Loaded config file (Parsed dictionary, see parse_config.py)
    :parsed_argument
    """
    import src.perceiver_pooling.utils as model_utils
    from src import data_utils
    from src.utils import save_model_configs

    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # model parameters
    model_parameters = parsed_config['model_parameters']
    learning_rate = model_parameters['learning_rate']
    epochs = model_parameters['epochs']

    # general logging information
    general_parameters = parsed_config['general']
    log_dir = os.path.join(general_parameters['log_dir'])
    use_tensorboard = general_parameters['use_tensorboard']
    save_best_model_only = general_parameters['save_best_model_only']

    # early stopping parameters
    es_parameters = parsed_config['early_stopping']
    early_stopping_active = es_parameters['active']

    # load parameters of modality perceivers
    perceiver_params = parsed_config['perceivers']
    all_perceiver_parameters = {k: perceiver_params[k]['model_parameters'] for k in perceiver_params}

    # data loader settings
    data_settings = parsed_config['data_settings']
    _batch_size = data_settings['data_loader']['batch_size']
    _generation_seed = data_settings['data_loader']['generation_seed']
    _data_load_type = data_settings['data_loader']['load_type']
    workers = 4
    pin_memory = True
    persistent_workers = True

    # set global seed for reproducibility
    torch.manual_seed(_generation_seed)
    torch.cuda.manual_seed(_generation_seed)

    train_dataset, val_dataset = data_utils.build_train_val_dynamic_bag_dataset(
        perceiver_params=perceiver_params,
        data_loader_information=data_settings['data_loader']
    )

    ################################################################
    # model settings
    ################################################################
    model_attributes = parsed_config['model_attributes']
    print(model_attributes)
    model = PerceiverPooling(
        perceivers_parameters=all_perceiver_parameters,
        general_model_parameters=model_attributes
    ).to(device=_device)

    print('Trainable parameter count:', get_model_trainable_parameter_count(model))

    loss_function = F.cross_entropy

    network_operator = NetworkOperator(
        model=model, model_name=PerceiverPooling.__name__, log_dir=log_dir
    )

    # save the configuration files
    save_model_configs(
        log_dir=network_operator.log_dir, main_config=parsed_config)

    # perceiver learning rate scheduler - make sure that the epochs fit
    optimizer = Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[28, 34, 38])

    ################################################################
    # Early stopping
    ################################################################
    early_stopping = None
    if early_stopping_active:
        early_stopping = EarlyStoppingByLoss(
            patience=es_parameters['patience'],
            delta=es_parameters['delta'])

    def train_data_loader() -> Iterable:
        """
        Training data loader (especially when dealing with dynamic input where modalities and instances can vary
        in number
        :return:
        """
        return BeautifyDataLoaderIterations(
            data_loader=create_data_loader(dataset=train_dataset, workers=workers, batch_size=1 if _data_load_type == 'dynamic' else _batch_size,
                                           persistent_workers=persistent_workers, pin_memory=pin_memory),
            tqdm_description='Iterating training dataset'
        )

    def val_data_loader() -> Iterable:
        """
        Validation data loader (especially when dealing with dynamic input where modalities and instances can vary
        in number
        :return:
        """
        return BeautifyDataLoaderIterations(
            data_loader=create_data_loader(dataset=val_dataset, workers=workers, batch_size=1 if _data_load_type == 'dynamic' else _batch_size,
                                           persistent_workers=persistent_workers, pin_memory=pin_memory),
            tqdm_description='Iterating validation dataset'
        )

    def model_input_pre_hook(batch: Dict[str, torch.Tensor], labels: torch.Tensor) -> \
            Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if _data_load_type == 'dynamic':
            for name in batch:
                d = torch.squeeze(batch[name], dim=0)
                batch[name] = d
            labels = torch.squeeze(labels, dim=0)

        return batch, labels

    def after_epoch_hook(train_info: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                         eval_info: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Hook to interact with operate function and epoch information
        :param train_info:
        :param eval_info:
        :return:
        """
        # we only use it to set the scheduler correctly
        lr_scheduler.step()

    def additional_loss_callback(tensorboard_writer: SummaryWriter, mode: str, epoch: int, losses: torch.Tensor,
                                 model_outputs: torch.Tensor, targets: torch.Tensor):
        kwargs = {'operator': network_operator, 'device': _device, 'batch_size': _batch_size,
                  'train_dataset': train_dataset, 'val_dataset': val_dataset,
                  'persistent_workers': persistent_workers, 'workers': workers, 'pin_memory': pin_memory}
        model_utils.additional_loss_callback(
            tensorboard_writer=tensorboard_writer, mode=mode, epoch=epoch, losses=losses,
            model_outputs=model_outputs, targets=targets, **kwargs
        )

    network_operator.operate(
        epochs=epochs, device=_device, train_data_loader=train_data_loader(), eval_data_loader=val_data_loader(),
        optimizer=optimizer, loss_fn=loss_function, model_input_pre_hook=model_input_pre_hook, optimizer_pre_hook=None,
        train_after_batch_hook=None, eval_after_batch_hook=None,
        after_epoch_hook=after_epoch_hook, use_tensorboard=use_tensorboard,
        tensorboard_hook=additional_loss_callback, save_best_only=save_best_model_only,
        loss_optimization='minimize', early_stopping=early_stopping
    )


if __name__ == '__main__':
    # see the configuration folder and the provided config.ini
    # You can change the configuration file or provide a new one and passing it the the args
    # ( see parse_arguments methods )

    # comment the next two lines and uncomment the third to use script input arguments instead of fixed config
    example_config_path = os.path.join(current_dir, 'configs', 'hopfield_pooling_static_loader_no_noise', 'config.ini')
    parsed_config = parse_arguments(config_file_path=example_config_path)

    # parsed_config = parse_arguments()

    run_network(
        parsed_config=parsed_config
    )
