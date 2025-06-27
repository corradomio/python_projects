    NeuralNet.__init__
        The base class covers more generic cases. Depending on your use
        case, you might want to use :class:`.NeuralNetClassifier` or
        :class:`.NeuralNetRegressor`.

        In addition to the parameters listed below, there are parameters
        with specific prefixes that are handled separately. To illustrate
        this, here is an example:

        >>> net = NeuralNet(
        ...    ...,
        ...    optimizer=torch.optimizer.SGD,
        ...    optimizer__momentum=0.95,
        ...)

        This way, when ``optimizer`` is initialized, :class:`.NeuralNet`
        will take care of setting the ``momentum`` parameter to 0.95.

        (Note that the double underscore notation in
        ``optimizer__momentum`` means that the parameter ``momentum``
        should be set on the object ``optimizer``. This is the same
        semantic as used by sklearn.)

        Furthermore, this allows to change those parameters later:

        ``net.set_params(optimizer__momentum=0.99)``

        This can be useful when you want to change certain parameters
        using a callback, when using the net in an sklearn grid search,
        etc.

        By default an :class:`.EpochTimer`, :class:`.BatchScoring` (for
        both training and validation datasets), and :class:`.PrintLog`
        callbacks are added for convenience.

        Parameters
        ----------
        module : torch module (class or instance)
          A PyTorch :class:`~torch.nn.Module`. In general, the
          uninstantiated class should be passed, although instantiated
          modules will also work.

        criterion : torch criterion (class)
          The uninitialized criterion (loss) used to optimize the
          module.

        optimizer : torch optim (class, default=torch.optim.SGD)
          The uninitialized optimizer (update rule) used to optimize the
          module

        lr : float (default=0.01)
          Learning rate passed to the optimizer. You may use ``lr`` instead
          of using ``optimizer__lr``, which would result in the same outcome.

        max_epochs : int (default=10)
          The number of epochs to train for each ``fit`` call. Note that you
          may keyboard-interrupt training at any time.

        batch_size : int (default=128)
          Mini-batch size. Use this instead of setting
          ``iterator_train__batch_size`` and ``iterator_test__batch_size``,
          which would result in the same outcome. If ``batch_size`` is -1,
          a single batch with all the data will be used during training
          and validation.

        iterator_train : torch DataLoader
          The default PyTorch :class:`~torch.utils.data.DataLoader` used for
          training data.

        iterator_valid : torch DataLoader
          The default PyTorch :class:`~torch.utils.data.DataLoader` used for
          validation and test data, i.e. during inference.

        dataset : torch Dataset (default=skorch.dataset.Dataset)
          The dataset is necessary for the incoming data to work with
          pytorch's ``DataLoader``. It has to implement the ``__len__`` and
          ``__getitem__`` methods. The provided dataset should be capable of
          dealing with a lot of data types out of the box, so only change
          this if your data is not supported. You should generally pass the
          uninitialized ``Dataset`` class and define additional arguments to
          X and y by prefixing them with ``dataset__``. It is also possible
          to pass an initialzed ``Dataset``, in which case no additional
          arguments may be passed.

        train_split : None or callable (default=skorch.dataset.ValidSplit(5))
          If ``None``, there is no train/validation split. Else, ``train_split``
          should be a function or callable that is called with X and y
          data and should return the tuple ``dataset_train, dataset_valid``.
          The validation data may be ``None``.

        callbacks : None, "disable", or list of Callback instances (default=None)
          Which callbacks to enable. There are three possible values:

          If ``callbacks=None``, only use default callbacks,
          those returned by ``get_default_callbacks``.

          If ``callbacks="disable"``, disable all callbacks, i.e. do not run
          any of the callbacks, not even the default callbacks.

          If ``callbacks`` is a list of callbacks, use those callbacks in
          addition to the default callbacks. Each callback should be an
          instance of :class:`.Callback`.

          Callback names are inferred from the class
          name. Name conflicts are resolved by appending a count suffix
          starting with 1, e.g. ``EpochScoring_1``. Alternatively,
          a tuple ``(name, callback)`` can be passed, where ``name``
          should be unique. Callbacks may or may not be instantiated.
          The callback name can be used to set parameters on specific
          callbacks (e.g., for the callback with name ``'print_log'``, use
          ``net.set_params(callbacks__print_log__keys_ignored=['epoch',
          'train_loss'])``).

        predict_nonlinearity : callable, None, or 'auto' (default='auto')
          The nonlinearity to be applied to the prediction. When set to
          'auto', infers the correct nonlinearity based on the criterion
          (softmax for :class:`~torch.nn.CrossEntropyLoss` and sigmoid for
          :class:`~torch.nn.BCEWithLogitsLoss`). If it cannot be inferred
          or if the parameter is None, just use the identity
          function. Don't pass a lambda function if you want the net to be
          pickleable.

          In case a callable is passed, it should accept the output of the
          module (the first output if there is more than one), which is a
          PyTorch tensor, and return the transformed PyTorch tensor.

          This can be useful, e.g., when
          :func:`~skorch.NeuralNetClassifier.predict_proba`
          should return probabilities but a criterion is used that does
          not expect probabilities. In that case, the module can return
          whatever is required by the criterion and the
          ``predict_nonlinearity`` transforms this output into
          probabilities.

          The nonlinearity is applied only when calling
          :func:`~skorch.classifier.NeuralNetClassifier.predict` or
          :func:`~skorch.classifier.NeuralNetClassifier.predict_proba` but
          not anywhere else -- notably, the loss is unaffected by this
          nonlinearity.

        warm_start : bool (default=False)
          Whether each fit call should lead to a re-initialization of the
          module (cold start) or whether the module should be trained
          further (warm start).

        verbose : int (default=1)
          This parameter controls how much print output is generated by
          the net and its callbacks. By setting this value to 0, e.g. the
          summary scores at the end of each epoch are no longer printed.
          This can be useful when running a hyperparameter search. The
          summary scores are always logged in the history attribute,
          regardless of the verbose setting.

        device : str, torch.device, or None (default='cpu')
          The compute device to be used. If set to 'cuda' in order to use
          GPU acceleration, data in torch tensors will be pushed to cuda
          tensors before being sent to the module. If set to None, then
          all compute devices will be left unmodified.

        compile : bool (default=False)
          If set to ``True``, compile all modules using ``torch.compile``. For this
          to work, the installed torch version has to support ``torch.compile``.
          Compiled modules should work identically to non-compiled modules but
          should run faster on new GPU architectures (Volta and Ampere for
          instance).
          Additional arguments for ``torch.compile`` can be passed using the dunder
          notation, e.g. when initializing the net with ``compile__dynamic=True``,
          ``torch.compile`` will be called with ``dynamic=True``.

        use_caching : bool or 'auto' (default='auto')
          Optionally override the caching behavior of scoring callbacks. Callbacks
          such as :class:`.EpochScoring` and :class:`.BatchScoring` allow to cache
          the inference call to save time when calculating scores during training at
          the expense of memory. In certain situations, e.g. when memory is tight,
          you may want to disable caching. As it is cumbersome to change the setting
          on each callback individually, this parameter allows to override their
          behavior globally.
          By default (``'auto'``), the callbacks will determine if caching is used
          or not. If this argument is set to ``False``, caching will be disabled on
          all callbacks. If set to ``True``, caching will be enabled on all
          callbacks.
          Implementation note: It is the job of the callbacks to honor this setting.

        torch_load_kwargs : dict or None (default=None)
          Additional arguments that will be passed to torch.load when load pickled
          parameters.

          In particular, this is important to because PyTorch will switch (probably
          in version 2.6.0) to only allow weights to be loaded for security reasons
          (i.e weights_only switches from False to True). As a consequence, loading
          pickled parameters may raise an error after upgrading torch because some
          types are used that are considered insecure. In skorch, we will also make
          that switch at the same time. To resolve the error, follow the
          instructions in the torch error message to designate the offending types
          as secure. Only do this if you trust the source of the file.

          If you want to keep loading non-weight types the same way as before,
          please pass:

              torch_load_kwargs={'weights_only': False}

          You should be aware that this is considered insecure and should only be
          used if you trust the source of the file. However, this does not introduce
          new insecurities, it rather corresponds to the status quo from before
          torch made the switch.

          Another way to avoid this issue is to pass use_safetensors=True when
          calling save_params and load_params. This avoid using pickle in favor of
          the safetensors format, which is secure by design.

        Attributes
        ----------
        prefixes_ : list of str
          Contains the prefixes to special parameters. E.g., since there
          is the ``'optimizer'`` prefix, it is possible to set parameters like
          so: ``NeuralNet(..., optimizer__momentum=0.95)``. Some prefixes are
          populated dynamically, based on what modules and criteria are defined.

        cuda_dependent_attributes_ : list of str
          Contains a list of all attribute prefixes whose values depend on a
          CUDA device. If a ``NeuralNet`` trained with a CUDA-enabled device
          is unpickled on a machine without CUDA or with CUDA disabled, the
          listed attributes are mapped to CPU.  Expand this list if you
          want to add other cuda-dependent attributes.

        initialized_ : bool
          Whether the :class:`.NeuralNet` was initialized.

        module_ : torch module (instance)
          The instantiated module.

        criterion_ : torch criterion (instance)
          The instantiated criterion.

        callbacks_ : list of tuples
          The complete (i.e. default and other), initialized callbacks, in
          a tuple with unique names.

        _modules : list of str
          List of names of all modules that are torch modules. This list is
          collected dynamically when the net is initialized. Typically, there is no
          reason for a user to modify this list.

        _criteria : list of str
          List of names of all criteria that are torch modules. This list is
          collected dynamically when the net is initialized. Typically, there is no
          reason for a user to modify this list.

        _optimizers : list of str
          List of names of all optimizers. This list is collected dynamically when
          the net is initialized. Typically, there is no reason for a user to modify
          this list.



    NeuralNet.fit(X,y=None)

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        y : target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

