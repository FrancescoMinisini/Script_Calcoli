# hook-ahrs.utils.wmm.py
from PyInstaller.utils.hooks import collect_data_files
datas = collect_data_files(
    "ahrs.utils", includes=["WMM2020/WMM.COF"]
)
