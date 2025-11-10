from xarp.settings import settings


def settings_file_authorization(user_id: str) -> bool:
    return settings.authorized and user_id not in settings.authorized
