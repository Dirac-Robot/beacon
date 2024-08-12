import io
import json
import pickle
import unittest

from beacon.adict import ADict

from copy import deepcopy as dcp


class ADictUnitTest(unittest.TestCase):
    def setUp(self):
        self.simple_dict = {"name": "John Doe", "age": 30, "city": "New York"}
        self.nested_dict = {
            "user": {
                "name": "John Doe",
                "age": 30,
                "address": {"city": "New York", "country": "USA"},
            },
            "posts": [{"title": "Post 1", "content": "Hello, world!"}],
            "family": ["mother", "father", "sister", "brother", "wife", "son"]
        }
        self.long_dict = {
            "user-0": {"name": "Michael", "score": 12},
            "user-1": {"name": "William", "score": 61},
            "user-2": {"name": "Wilson", "score": 52},
            "user-3": {"name": "Andrew", "score": 93},
            "user-4": {"name": "Eugene", "score": 28},
            "user-5": {"name": "Richard", "score": 42},
            "user-6": {"name": "Lucy", "score": 66},
            "user-7": {"name": "Tracy", "score": 77},
            "user-8": {"name": "John", "score": 78},
            "user-9": {"name": "Elly", "score": 100}
        }
        self.adict_simple = ADict(self.simple_dict)
        self.adict_nested = ADict(self.nested_dict)
        self.adict_long = ADict(self.long_dict)

    def test_initialize(self):
        ADict(self.simple_dict)
        ADict(self.nested_dict)

    def test_default(self):
        def auto_nested_config():
            return ADict(default=auto_nested_config)
        config = auto_nested_config()
        config.plan.alpha.beta.gamma = 0
        self.assertEqual(config.plan.alpha.beta.gamma, 0)
        self.assertIsInstance(config.plan, ADict)
        self.assertIsInstance(config.plan.alpha, ADict)
        self.assertIsInstance(config.plan.alpha.beta, ADict)

    def test_initialize_from_kwargs(self):
        ADict(**self.simple_dict)
        ADict(**self.nested_dict)

    def test_implicit_convert(self):
        config = ADict(self.nested_dict)
        self.assertIsInstance(config['user'], ADict)
        self.assertIsInstance(config['user']['address'], ADict)

    def test_get_item_by_attribute(self):
        self.assertEqual(self.adict_simple.age, 30)

    def test_get_item_by_key(self):
        self.assertEqual(self.adict_simple['city'], "New York")

    def test_set_item_by_attribute(self):
        self.adict_simple.age = 10
        self.adict_nested.user.age = 12
        self.assertEqual(self.adict_simple.age, 10)
        self.assertEqual(self.adict_nested.user.age, 12)

    def test_set_item_by_key(self):
        self.adict_simple["city"] = "London"
        self.adict_nested.user.address["city"] = "London"
        self.assertEqual(self.adict_simple.city, "London")
        self.assertEqual(self.adict_nested.user.address.city, "London")

    def test_get_item_by_iterable(self):
        self.assertEqual(
            self.adict_nested.user["name", "age"],
            ["John Doe", 30]
        )

    def test_compute_with_value(self):
        self.adict_simple.age *= 10
        self.assertEqual(
            self.adict_simple.age, 300
        )

    def test_set_item_by_iterable(self):
        self.adict_nested.user["name", "age", "address"] = ["Richard Kim", 29, {"city": "Seoul", "country": "Korea"}]
        self.assertEqual(
            list(self.adict_nested.user.values()),
            ["Richard Kim", 29, ADict({"city": "Seoul", "country": "Korea"})]
        )

    def test_delete(self):
        with self.assertRaises(KeyError):
            del self.adict_simple["gender"]
            del self.adict_nested["careers"]
        del self.adict_simple["age"]
        self.assertTrue("age" not in self.adict_simple)

    def test_construct_with_various_inputs(self):
        with self.assertRaises(TypeError):
            ADict(10)
            ADict(self.nested_dict["family"])
            ADict(None)

    def test_clear(self):
        self.adict_simple.clear()
        self.assertEqual(len(self.adict_simple), 0)

    def test_deepcopy(self):
        self.assertEqual(self.adict_nested, dcp(self.adict_nested))

    def test_update(self):
        adict_nested = dcp(self.adict_nested)
        adict_nested.user.update(self.adict_simple)
        self.assertEqual(
            dict(adict_nested),
            {
                "user": {
                    "name": "John Doe",
                    "age": 30,
                    "city": "New York",
                    "address": {"city": "New York", "country": "USA"},
                },
                "posts": [{"title": "Post 1", "content": "Hello, world!"}],
                "family": ["mother", "father", "sister", "brother", "wife", "son"]
            }
        )

    def test_convert_between_json(self):
        adict_json = self.adict_nested.json()
        restored_adict = ADict(json.loads(adict_json))
        self.assertEqual(self.adict_nested, restored_adict)

    def test_convert_to_structural_repr(self):
        structural_repr = self.adict_nested.get_structural_repr()
        self.adict_nested.user.age = 31  # type is not changed
        edited_structural_repr = self.adict_nested.get_structural_repr()
        self.adict_nested.user.age = '35'  # type is changed
        type_edited_structural_repr = self.adict_nested.get_structural_repr()
        self.assertEqual(structural_repr, edited_structural_repr)
        self.assertNotEqual(structural_repr, type_edited_structural_repr)

    def test_convert_to_structural_hash(self):
        structural_hash = self.adict_nested.get_structural_hash()
        self.adict_nested.user.age = 31  # type is not changed
        edited_structural_hash = self.adict_nested.get_structural_hash()
        self.adict_nested.user.age = '35'  # type is changed
        type_edited_structural_hash = self.adict_nested.get_structural_hash()
        self.assertEqual(structural_hash, edited_structural_hash)
        self.assertNotEqual(structural_hash, type_edited_structural_hash)

    def test_pickle(self):
        pickle_io = io.BytesIO()
        pickle.dump(dcp(self.adict_nested), pickle_io)
        pickle_io.seek(0)
        restored_adict = pickle.load(pickle_io)
        self.assertEqual(self.adict_nested, restored_adict)

    def test_raw(self):
        self.assertEqual(self.adict_simple.raw('name'), ADict(key='name', value='John Doe'))

    def test_convert_to_immutable(self):
        self.adict_simple.convert_to_immutable()
        self.adict_nested.convert_to_immutable()
        with self.assertRaises(TypeError):
            del self.adict_simple['name']
        with self.assertRaises(TypeError):
            self.adict_simple['name'] = 'poo'
        with self.assertRaises(TypeError):
            del self.adict_nested['user']
        with self.assertRaises(TypeError):
            self.adict_nested['user'] = 'poo'
        with self.assertRaises(TypeError):
            del self.adict_simple.name
        with self.assertRaises(TypeError):
            self.adict_simple.name = 'poo'
        with self.assertRaises(TypeError):
            del self.adict_nested.user
        with self.assertRaises(TypeError):
            self.adict_nested.user = 'poo'

    def test_replace_keys(self):
        pass

    def test_recurrent_update(self):
        self.adict_nested.update(
            {
                "user": {
                    "name": "John Christopher",
                    "address": {"city": "Texas"}
                },
                "posts": [{"title": "Post 3", "content": "Hello, world!"}]
            },
            user={'age': 20},
            recurrent=True
        )
        self.assertIn('country', self.adict_nested.user.address)
        self.assertEqual(self.adict_nested.user.address.city, 'Texas')

    def test_convert_from_iterables(self):
        adict_converted = ADict([('Andrew', 'Jackson'), ('John', 'Christopher')])
        self.assertEqual(adict_converted.Andrew, 'Jackson')
        adict_converted['aa'] = ADict(bb=ADict(ee='ll'))


if __name__ == "__main__":
    unittest.main()
