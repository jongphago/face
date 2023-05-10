from math import floor


def get_folder_name(family_id: str) -> str:
    """_summary_

    Args:
        family_id (str): 접두사 F를 포함하는 네자리 숫자 문자열, 'F0###' 형식

    Returns:
        str: prefix를 제거한 네자리 숫자 문자열. 동일한 백의 자리를 같는 수 중
        가장 작은 자연수. '0#00' 형식
    """
    return f"{floor(int(family_id[-4:])/100)*100:04d}"
