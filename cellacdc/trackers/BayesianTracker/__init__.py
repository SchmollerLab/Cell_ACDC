try:
    import btrack
except ModuleNotFoundError:
    pkg_name = 'BayesianTracker'
    from PyQt5.QtWidgets import QMessageBox
    txt = (
        f'Cell-ACDC is going to download and install "{pkg_name}".\n\n'
        'Make sure you have an active internet connection, '
        'before continuing. '
        'Progress will be displayed on the terminal\n\n'
        'Alternatively, you can cancel the process and try later.'
    )
    answer = msg.information(
        None, f'Install {pkg_name}', txt, msg.Ok | msg.Cancel
    )
    if answer == msg.Cancel:
        raise ModuleNotFoundError(
            f'User aborted {pkg_name} installation'
        )
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'btrack']
    )
