import unittest

from beacon.halo import HALo


class HaLoUnitTest(unittest.TestCase):
    def setUp(self):
        self.iterable = range(30)
        self.halo = HALo(range(30))
        self.halo_2nd = HALo(range(30))

    def test_iterate(self):
        for item in self.halo:
            if self.halo.first:
                self.assertEqual(item, 0)
            if self.halo.last:
                self.assertEqual(item, 29)

    def test_curl(self):
        for _ in self.halo:
            self.assertEqual(HALo.curl(), self.halo)
            for _ in self.halo_2nd:
                self.assertEqual(HALo.curl(), self.halo_2nd)
            self.assertEqual(HALo.curl(), self.halo)


if __name__ == "__main__":
    unittest.main()
