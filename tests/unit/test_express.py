import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from xarp.commands import Bundle
from xarp.commands.entities import CreateOrUpdateElementCommand
from xarp.entities import Element
from xarp.express import AsyncXR, _coerce_elements


class TestCoerceElements(unittest.TestCase):
    def test_single_element_returns_one_element_list(self):
        element = Element(key="one")

        self.assertEqual(_coerce_elements(element), [element])

    def test_list_returns_list_of_elements(self):
        elements = [Element(key="one"), Element(key="two")]

        self.assertEqual(_coerce_elements(elements), elements)

    def test_tuple_returns_list_of_elements(self):
        first = Element(key="one")
        second = Element(key="two")

        self.assertEqual(_coerce_elements((first, second)), [first, second])

    def test_generator_is_materialized_once(self):
        yielded = []

        def generate():
            for key in ("one", "two"):
                yielded.append(key)
                yield Element(key=key)

        elements = _coerce_elements(generate())

        self.assertEqual([element.key for element in elements], ["one", "two"])
        self.assertEqual(yielded, ["one", "two"])

    def test_empty_iterable_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "at least one element"):
            _coerce_elements([])


class TestAsyncXRUpdate(unittest.IsolatedAsyncioTestCase):
    async def _update_and_get_command(self, element):
        remote = SimpleNamespace()
        remote.execute = AsyncMock(return_value=SimpleNamespace(value=[None]))
        xr = AsyncXR(remote)

        await xr.update(element)

        bundle = remote.execute.call_args.args[0]
        self.assertIsInstance(bundle, Bundle)
        command = bundle.cmds[0]
        self.assertIsInstance(command, CreateOrUpdateElementCommand)
        return command

    async def test_update_accepts_single_element(self):
        element = Element(key="one")

        command = await self._update_and_get_command(element)

        self.assertEqual(command.elements, [element])

    async def test_update_accepts_list_of_elements(self):
        elements = [Element(key="one"), Element(key="two")]

        command = await self._update_and_get_command(elements)

        self.assertEqual(command.elements, elements)

    async def test_update_accepts_tuple_of_elements(self):
        elements = (Element(key="one"), Element(key="two"))

        command = await self._update_and_get_command(elements)

        self.assertEqual(command.elements, list(elements))

    async def test_update_accepts_generator_of_elements(self):
        elements = (Element(key=key) for key in ("one", "two"))

        command = await self._update_and_get_command(elements)

        self.assertEqual([element.key for element in command.elements], ["one", "two"])

    async def test_update_rejects_empty_iterable(self):
        remote = SimpleNamespace()
        remote.execute = AsyncMock(return_value=SimpleNamespace(value=[None]))
        xr = AsyncXR(remote)

        with self.assertRaisesRegex(ValueError, "at least one element"):
            await xr.update([])

        remote.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
