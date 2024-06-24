from stdlib.dictk import dictk


def main():
    ts = dictk(keys=['window_length', 'prediction_length'])

    data = dictk(keys=['scaler_type'])

    mlp_model = dictk(keys=['n_layers', 'hidden_size'])

    rnn_model = dictk(keys=['inference_input_size', 'encoder_n_layers', 'encoder_hidden_size', 'encoder_activation',
                            'encoder_bias', 'encoder_dropout', 'context_size', 'decoder_hidden_size',
                            'decoder_layers'])

    engine = dictk(keys=['loss', 'loss_kwargs', 'valid_loss', 'valid_loss_kwargs',
                         'optimizer', 'optimizer_kwargs', 'lr_scheduler', 'lr_scheduler_kwargs'])

    trainer = dictk(keys=['max_steps', 'learning_rate', 'num_lr_decays', 'early_stop_patience_steps',
                          'val_check_steps', 'batch_size', 'valid_batch_size', 'windows_batch_size',
                          'inference_windows_batch_size', 'start_padding_enabled', 'step_size',
                          'exclude_insample_y', 'random_seed', 'num_workers_loader', 'drop_last_loader'])

    other = dictk(keys=['alias'])
    other['alias'] = "ciccio"
    other.alias += "pasticcio"

    print(other['alias'])
    # print(other.alias)
    pass


if __name__ == "__main__":
    main()
