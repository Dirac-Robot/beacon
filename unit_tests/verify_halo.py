import unittest

from beacon.halo import HaLo


class HaLoUnitTest(unittest.TestCase):
    def setUp(self):
        self.iterable = range(30)
        self.halo = HaLo(range(30))
        self.halo_2nd = HaLo(range(30))

    def test_iterate(self):
        for item in self.halo:
            if self.halo.first:
                self.assertEqual(item, 0)
            if self.halo.last:
                self.assertEqual(item, 29)

    def test_curl(self):
        for _ in self.halo:
            self.assertEqual(HaLo.curl(), self.halo)
            for _ in self.halo_2nd:
                self.assertEqual(HaLo.curl(), self.halo_2nd)
            self.assertEqual(HaLo.curl(), self.halo)


if __name__ == "__main__":
    unittest.main()
