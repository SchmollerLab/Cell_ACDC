from . import widgets, html_utils
from . import issues_url

def warnTooManyItems(mainWin, numItems, qparent):
    mainWin.logger.info(
        '[WARNING]: asking user what to do with too many graphical items...'
    )
    msg = widgets.myMessageBox()
    txt = html_utils.paragraph(f"""
        You loaded a segmentation mask that has <b>{numItems} objects</b>.<br><br>
        Creating <b>high resolution</b> text annotations 
        for these many objects could take a <b>long time</b>.<br><br>
        We recommend <b>switching to low resolution</b> annotations.<br><br>
        You can still try to switch to high resolution later.<br><br>
        What do you want to do?
    """)

    _, stayHighResButton, switchToLowResButton = msg.warning(
        qparent, 'Too many objects', txt,
        buttonsTexts=(
            'Cancel', 'Stay on high resolution', 
            widgets.reloadPushButton(' Switch to low resolution ')              
        )
    )
    return msg.cancel, msg.clickedButton==switchToLowResButton

def warnRestartCellACDCcolorModeToggled(scheme, app_name='Cell-ACDC', parent=None):
    msg = widgets.myMessageBox(wrapText=False)
    txt = (
        'In order for the change to take effect, '
        f'<b>please restart {app_name}</b>'
    )
    if scheme == 'dark':
        issues_href = f'<a href="{issues_url}">GitHub page</a>'
        note_txt = (
            'NOTE: <b>dark mode</b> is a recent feature so if you see '
            'if you see anything odd,<br>'
            'please, <b>report it</b> by opening an issue '
            f'on our {issues_href}.<br><br>'
            'Thanks!'
        )
        txt = f'{txt}<br><br>{note_txt}'
    txt = html_utils.paragraph(txt)
    msg.information(parent, f'Restart {app_name}', txt)