from . import issues_url

def warnTooManyItems(mainWin, numItems, qparent):
    from . import widgets, html_utils
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
    from . import widgets, html_utils
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

class DataTypeWarning(RuntimeWarning):
    def __init__(self, message):
        self._message = message
    
    def __str__(self):
        return repr(self._message)

def warn_image_overflow_dtype(input_dtype, max_value, inferred_dtype):
    import warnings
    warnings.warn(
        f'The input image has data type {input_dtype}. Since it is neither '
        f'8-bit, 16-bit, nor 32-bit the data was inferred as {inferred_dtype} '
        f'from the max value of the image of {max_value}.', 
        DataTypeWarning
    )

def warn_cca_integrity(self, txt, category):
    from . import html_utils, widgets
    from qtpy.QtWidgets import QCheckBox
    
    preamble = html_utils.paragraph(
        'WARNING: <b>integrity of cell cycle annotations</b> '
        'might be <b>compromised</b>:'
    )
    
    msg_text = f'{preamble}{txt}'
    
    stopSpecificMessageCheckbox = QCheckBox(
        'Stop warning with this specific message'
    )
    stopCategoryCheckbox = QCheckBox(
        f'Stop warning about "{category}"'
    )
    disableAllWarningsCheckbox = QCheckBox(
        'Disable all warnings'
    )
    
    checkboxes = (
        stopSpecificMessageCheckbox, 
        stopCategoryCheckbox, 
        disableAllWarningsCheckbox
    )
    
    msg = widgets.myMessageBox(wrapText=False)
    msg.warning(
        self, 'Annotations integrity warning', msg_text, 
        widgets=checkboxes
    )
    if stopSpecificMessageCheckbox.isChecked():
        return txt
    
    if stopCategoryCheckbox.isChecked():
        return category
    
    if disableAllWarningsCheckbox.isChecked():
        return 'disable_all'
    
    return ''