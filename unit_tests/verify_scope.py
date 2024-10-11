import unittest
import sys
from itertools import chain

from beacon.adict import ADict
from beacon.scope import Scope, parse_args_pythonic


class ScopeUnitTest(unittest.TestCase):
    def setUp(self):
        Scope.initialize_registry()
        config = ADict(
            learning_rate=0.1,
            batch_size=128
        )
        scope = Scope(config=config, name='unit_test_config')
        self.config = config
        self.scope = scope
        sys.argv = []

    def test_decorate(self):
        scope = self.scope

        @scope
        def test_func(unit_test_config):
            return unit_test_config.learning_rate

        self.assertEqual(test_func(), 0.1)

    def test_observe(self):
        scope = self.scope

        @scope.observe()
        def trivial_config(unit_test_config):
            unit_test_config.learning_rate = 0.05

        scope.assign('trivial_config')
        scope.apply()
        self.assertEqual(self.config.learning_rate, 0.05)

    def test_priority(self):
        scope = self.scope

        @scope.observe(priority=1)
        def view_1(unit_test_config):
            unit_test_config.learning_rate = 0.05

        @scope.observe(priority=10)
        def view_2(unit_test_config):
            unit_test_config.learning_rate = 0.5
            unit_test_config.batch_size = 256

        scope.assign('batch_size=1024')
        scope.assign('view_2')
        scope.assign('view_1')
        scope.apply()
        self.assertEqual(self.config.learning_rate, 0.5)
        self.assertEqual(self.config.batch_size, 1024)

    def test_parsing(self):
        scope = self.scope

        @scope.observe(priority=10)
        def test_view(unit_test_config):
            unit_test_config.learning_rate = 0.05

        sys.argv = (
            'test.py '
            'batch_size=1024 '
            'num_layers=[1, 4, [1, 2, 3]] '
            'test_view '
            'prompt=`Elsa is doing magic.` '
            'equation=`a=\`0.1, b=[1, 2, 3], mode=\`train\`\``'
        ).split()
        parse_args_pythonic()
        self.scope.apply()
        self.assertEqual(self.config.learning_rate, 0.05)
        self.assertEqual(self.config.batch_size, 1024)
        self.assertEqual(self.config.num_layers, [1, 4, [1, 2, 3]])
        self.assertEqual(self.config.prompt, 'Elsa is doing magic.')
        self.assertEqual(self.config.equation, 'a="0.1, b=[1, 2, 3], mode="train""')

    def test_positional_case(self):
        scope = self.scope

        @scope
        def test_front(my_argument, unit_test_config):
            unit_test_config.check_argument = my_argument

        @scope
        def test_back(unit_test_config, my_argument):
            unit_test_config.check_argument = my_argument

        test_front('hi')
        self.assertEqual(self.config.check_argument, 'hi')
        test_back('bye')
        self.assertEqual(self.config.check_argument, 'bye')

    def test_lazy_context(self):
        scope = self.scope

        @scope.observe()
        def test_view(unit_test_config):
            unit_test_config.learning_rate = 0.1
            unit_test_config.factor = 1
            with Scope.lazy():
                if unit_test_config.learning_rate == 0.1:
                    unit_test_config.batch_size = 256*unit_test_config.factor
                else:
                    unit_test_config.batch_size = 1024

        sys.argv = 'test.py learning_rate=0.1 test_view prompt=`Elsa is doing magic.` eps=[1, 2] factor=2'.split()
        parse_args_pythonic()
        self.scope.apply()
        self.assertEqual(self.config.learning_rate, 0.1)
        self.assertEqual(self.config.batch_size, 512)
        self.assertEqual(self.config.prompt, 'Elsa is doing magic.')
        self.assertEqual(self.config.eps, [1, 2])

    def test_context_with_compile(self):
        scope = self.scope

        @scope.observe()
        def test_view(unit_test_config):
            unit_test_config.learning_rate = 0.1
            unit_test_config.factor = 1
            with Scope.lazy(with_compile=True):
                if unit_test_config.learning_rate == 0.1:
                    unit_test_config.batch_size = 256*unit_test_config.factor
                else:
                    unit_test_config.batch_size = 1024

        sys.argv = 'test.py learning_rate=0.1 test_view prompt=`Elsa is doing magic.` eps=[1, 2] factor=2'.split()
        parse_args_pythonic()
        self.scope.apply()
        self.assertEqual(self.config.learning_rate, 0.1)
        self.assertEqual(self.config.batch_size, 512)
        self.assertEqual(self.config.prompt, 'Elsa is doing magic.')
        self.assertEqual(self.config.eps, [1, 2])

    def test_view_chaining(self):
        scope = self.scope

        @scope.observe()
        def test_view(unit_test_config):
            unit_test_config.learning_rate = 0.1
            unit_test_config.factor = 1
            with Scope.lazy(with_compile=True):
                if unit_test_config.learning_rate == 0.1:
                    unit_test_config.batch_size = 256*unit_test_config.factor
                else:
                    unit_test_config.batch_size = 1024

        @scope.observe(chain_with='test_view')
        def test_chained_view(unit_test_config):
            if unit_test_config.learning_rate == 0.1:
                unit_test_config.weight_decay = 1
            else:
                unit_test_config.weight_decay = 0
            unit_test_config.learning_rate = 0.2

        sys.argv = 'test.py test_chained_view learning_rate=0.1 factor=2'.split()
        parse_args_pythonic()
        self.scope.apply()
        self.assertEqual(self.config.learning_rate, 0.2)
        self.assertEqual(self.config.batch_size, 1024)
        self.assertEqual(self.config.weight_decay, 1)

    def test_deeper_view_chaining(self):
        scope = self.scope

        @scope.observe()
        def test_view(unit_test_config):
            unit_test_config.learning_rate = 0.1
            unit_test_config.factor = 1
            with Scope.lazy(with_compile=True):
                if unit_test_config.learning_rate == 0.1:
                    unit_test_config.batch_size = 256*unit_test_config.factor
                else:
                    unit_test_config.batch_size = 1024

        @scope.observe(chain_with='test_view')
        def test_chained_view(unit_test_config):
            if unit_test_config.learning_rate == 0.1:
                unit_test_config.weight_decay = 1
            else:
                unit_test_config.weight_decay = 0
            unit_test_config.learning_rate = 0.2

        @scope.observe(chain_with='test_chained_view')
        def test_chained_chained_view(unit_test_config):
            pass

        sys.argv = 'test.py test_chained_chained_view learning_rate=0.1 factor=2 model_name=`resnet`'.split()
        parse_args_pythonic()
        self.scope.apply()
        self.assertEqual(self.config.learning_rate, 0.2)
        self.assertEqual(self.config.batch_size, 1024)
        self.assertEqual(self.config.weight_decay, 1)
        self.assertEqual(self.config.model_name, 'resnet')

    def test_activate_and_pause(self):
        scope = self.scope

        @scope.observe()
        def test_view(unit_test_config):
            unit_test_config.learning_rate = 0.1
            unit_test_config.factor = 1

        @scope
        def test_main(unit_test_config=None):
            return unit_test_config

        with scope.pause():
            pause_result = test_main()
        self.assertIsNone(pause_result)
        with self.assertRaises(TypeError):
            test_main(dict())
        self.assertIsNotNone(test_main())

    def test_get_assigned_views(self):
        scope = self.scope

        @scope.observe()
        def test_view(unit_test_config):
            unit_test_config.learning_rate = 0.1
            unit_test_config.factor = 1

        @scope.observe(chain_with='test_view')
        def test_chained_view(unit_test_config):
            if unit_test_config.learning_rate == 0.1:
                unit_test_config.weight_decay = 1
            else:
                unit_test_config.weight_decay = 0
            unit_test_config.learning_rate = 0.2

        @scope.observe(chain_with='test_chained_view')
        def test_chained_chained_view(unit_test_config):
            pass

        sys.argv = 'test.py test_chained_chained_view learning_rate=0.1 factor=2'.split()
        parse_args_pythonic()
        self.scope.apply()
        self.assertEqual(
            '-'.join(chain(*self.scope.get_assigned_views().values())),
            'test_view-test_chained_view-test_chained_chained_view-learning_rate=0.1-factor=2'
        )


if __name__ == "__main__":
    unittest.main()
