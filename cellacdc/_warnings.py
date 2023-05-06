from . import widgets, html_utils

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