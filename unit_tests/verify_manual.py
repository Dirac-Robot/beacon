from beacon.scope import Scope, MultiScope

scope_1 = Scope(name='config_1')
scope_2 = Scope(name='config_2')
scope = MultiScope(scope_1, scope_2)


@scope_1.observe(default=True)
def default_1(config_1):
    config_1.lr = 0.1


@scope_1.manual
def default_manual_1(config_1):
    config_1.lr = 'learning rate.'


@scope_2.observe(default=True)
def default_2(config_2):
    config_2.batch_size = 16


@scope_2.observe(default=True, lazy=True)
def lazy_view(config_2):
    config_2.batch_size = 16


@scope_2.manual
def default_manual_2(config_2):
    config_2.batch_size = 'batch size.'


@scope
def main(config_1, config_2):
    pass


if __name__ == '__main__':
    import sys
    sys.argv.insert(1, 'manual')
    main()
