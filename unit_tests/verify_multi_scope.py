from beacon.scope import Scope, MultiScope


scope_1 = Scope(name='config_1')
scope_2 = Scope(name='config_2')
multi_scope = MultiScope(scope_1, scope_2)


@scope_1.observe(default=True)
def default_1(config_1):
    config_1.text = 'Hello'


@scope_2.observe(default=True)
def default_2(config_2):
    config_2.text = 'hello'


@multi_scope
def main(config_1, config_2):
    print(config_1, config_2)


if __name__ == '__main__':
    main()
