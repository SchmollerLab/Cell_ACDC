import os
from functools import partial
import re

from cellacdc import html_utils, myutils

from . import issues_url
from . import urls
from . import error_below, error_close

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

def warn_cca_integrity(txt, category, qparent, go_to_frame_callback=None):
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
    if go_to_frame_callback is not None and txt.find('At frame n.') != -1:
        frame_n = re.findall(r'At frame n. (\d+)', txt)[0]
        goToFrameButton = widgets.NavigatePushButton(f'Go to frame n. {frame_n}')
        goToFrameButton = msg.addButton(goToFrameButton)
        goToFrameButton.disconnect()
        goToFrameButton.clicked.connect(
            partial(go_to_frame_callback, int(frame_n))
        )
        
    msg.warning(
        qparent, 'Annotations integrity warning', msg_text, 
        widgets=checkboxes
    )
    
    if stopSpecificMessageCheckbox.isChecked():
        return txt
    
    if stopCategoryCheckbox.isChecked():
        return category
    
    if disableAllWarningsCheckbox.isChecked():
        return 'disable_all'
    
    return ''

def warn_installing_different_cellpose_version(
        requested_version, installed_version
    ):
    from cellacdc import widgets
    if not myutils.is_gui_running():
        print(
            f'[WARNING]: You requested to install `Cellpose {requested_version}` '
            f'but you already have `Cellpose {installed_version}`.\n\n'
            f'If you proceed, Cell-ACDC will *uninstall* `{installed_version}` ' 
            f'and will install `{requested_version}`.'
        )
        return False
    
    note_text = """
    You can still proceed and let Cell-ACDC take care of 
    uninstalling/installing the right versions every time you request it.
    """
    txt = html_utils.paragraph(f"""
        [WARNING]: You requested to install 
        <code>Cellpose {requested_version}</code>, however you <b>already have</b> 
        <code>Cellpose {installed_version}</code>.<br><br>
        If you proceed, Cell-ACDC will <b>uninstall</b> <code>{installed_version}</code> 
        and will install <code>{requested_version}</code>.<br><br>
        If you plan to use both versions, we recommend installing Cell-ACDC 
        again in a <b>different environment</b><br> where you can keep one 
        version of Cellpose per environment.<br><br>
        Thank you for your patience!<br>
        {html_utils.to_note(note_text)}
    """)
    msg = widgets.myMessageBox(wrapText=False)
    msg.warning(
        None, 'Cellpose already installed', txt, 
        buttonsTexts=('Cancel', 'Ok')
    )
    return msg.cancel

def warn_download_bioformats_jar_failed(jar_dst_filepath, qparent=None):
    from cellacdc import widgets
    href = html_utils.href_tag('here', urls.bioformats_download_page)
    txt = html_utils.paragraph(f"""
        [WARNING]: <b>Download of <code>bioformats_package.jar</code> failed.
        </b><br><br>
        
        Please, download it from {href}, and place the jar file here:
    """)
    msg = widgets.myMessageBox(wrapText=False)
    msg.warning(
        qparent, 'Download of bioformats failed', txt, 
        commands=(jar_dst_filepath,), 
        path_to_browse=os.path.dirname(jar_dst_filepath)
    )
    return msg.cancel

def warnNotEnoughG1Cells(numCellsG1, frame_i, numNewCells, qparent=None):
    from cellacdc import widgets
    if numCellsG1 == 0:
        G1_text = 'no cells'
    else:
        G1_text = f'only {numCellsG1} cells'
    text = html_utils.paragraph(
        f'In the next frame <b>{numNewCells} new object(s)</b> will '
        'appear (highlighted in green on left image).<br><br>'
        
        f'However, in the previous frame (frame n. {frame_i}) there are '
        f'<b>{G1_text}</b> in G1 available.<br><br>'
        
        'Note that cells <b>must be in G1 in the previous frame too</b>, '
        'because if they are in G1<br>'
        'only at current frame, assigning a bud to it would result in no '
        'G1 phase at all between current<br>'
        'and previous cell cycle.<br>'
        
        'You can either cancel the operation and annotate division on previous '
        'frames or continue.<br><br>'
        
        'If you continue the <b>new cell</b> will be annotated as a '
        '<b>cell in G1 with unknown history</b>.<br><br>'
        
        'Do you want to continue?<br>'
    )
    
    msg = widgets.myMessageBox(wrapText=False)
    _, yesButton = msg.warning(
        qparent, 'No cells in G1!', text, 
        buttonsTexts=('Cancel', 'Continue anyway (new cells will start in G1)')
    )
    return msg.clickedButton == yesButton
    

def log_pytorch_not_installed():
    print(error_below)
    print(
        'PyTorch is not installed. See here how to install it '
        f'{urls.install_pytorch}'
    )
    print(error_close)

def warnExportToVideo(qparent=None):
    from cellacdc import widgets
    txt = html_utils.paragraph(f"""
        Exporting to video will start now.<br><br>
        During this process, the GUI will <b>automatically update the images</b> 
        to save the video frames.<br><br>
        Please, <b>do not close the window during this process</b>, thanks!
    """)
    msg = widgets.myMessageBox(wrapText=False)
    msg.warning(
        qparent, 'Export to video is starting', txt, 
        buttonsTexts=('Cancel', 'Ok')
    )
    return msg.cancel

def warnDivisionAnnotationCannotBeUndone(ID, relID, issue_frame_i, qparent=None):
    from cellacdc import widgets
    txt = html_utils.paragraph(f"""
        Cell division annotation <b>cannot be undone</b> because Cell ID {relID} 
        is in 'S' phase at frame n. {issue_frame_i+1}.<br><br>
        By undoing division annotation, Cell ID {relID} would be restored as 
        relative of Cell ID {ID}, but this cannot be done.<br><br>
        The only solution is to go to frame n. {issue_frame_i} and reset the 
        annotations there.<br><br>
        Thank you for your patience! 
    """)
    msg = widgets.myMessageBox(wrapText=False)
    msg.warning(
        qparent, 'Division annotation cannot be undone', txt
    )
    return msg.cancel

def warnCannotAddRemovePointsProjection(qparent=None):
    from cellacdc import widgets
    txt = html_utils.paragraph(f"""
        Points <b>cannot be added or removed in a projection</b>!<br><br>
        Please, switch to "single z-slice" mode (bottom of the image on 
        the right of the z-slice scrollbar).<br><br>
        Thank you for your patience.
    """)
    msg = widgets.myMessageBox(wrapText=False)
    msg.warning(qparent, 'WARNING: Editing points in projection', txt)