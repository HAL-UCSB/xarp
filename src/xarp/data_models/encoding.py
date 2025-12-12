import base64
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema


class LazyBase64Bytes(bytes):
    """
     behaves like `bytes` in memory and encodes to base64 string
    """

    @classmethod
    def __get_pydantic_core_schema__(
            cls,
            source_type: Any,
            handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # validator: accept bytes/bytearray or base64 string
        def validate(value: Any) -> 'LazyBase64Bytes':
            if isinstance(value, cls):
                return value

            if isinstance(value, (bytes, bytearray)):
                return cls(value)

            if isinstance(value, str):
                try:
                    decoded = base64.b64decode(value)
                except Exception as e:
                    # Let pydantic wrap this as a validation error
                    raise ValueError(f'Invalid base64 string: {e}') from e
                return cls(decoded)

            raise TypeError(
                f'LazyBase64Bytes: expected bytes, bytearray or base64 string, '
                f'got {type(value).__name__}'
            )

        # serializer: for JSON output -> base64 str; for python -> raw bytes
        def serialize(value: 'LazyBase64Bytes', info: core_schema.SerializationInfo):
            # info.mode is 'python' or 'json' :contentReference[oaicite:1]{index=1}
            if info.mode == 'json':
                return base64.b64encode(bytes(value)).decode('ascii')
            return bytes(value)

        return core_schema.no_info_before_validator_function(
            validate,
            schema=core_schema.bytes_schema(
                serialization=core_schema.plain_serializer_function_ser_schema(
                    serialize,
                    info_arg=True,
                    when_used='always',
                )
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
            cls,
            schema: core_schema.CoreSchema,
            handler,
    ) -> JsonSchemaValue:
        json_schema = handler(schema)
        json_schema.update(type='string', format='byte')
        return json_schema
