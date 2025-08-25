from xarp import settings


def is_authorized(user: str) -> bool:
    with open(settings.authorization_file_path) as file:
        authorized = file.read()
        return user in authorized
