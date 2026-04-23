from functools import partial

from qtpy.QtCore import (
    QTimer,
)

from cellacdc import printl

def q_debug(*args, **kwargs):
    guiWIn = args[0]
    
    opacities = {
        'img1': guiWIn.img1.opacity(),
    }
    
    items_low_opacity = []
    items_high_opacity = []
    
    if opacities['img1'] < 0.01:
        items_low_opacity.append(guiWIn.img1)
    elif opacities['img1'] > 0.99:
        items_high_opacity.append(guiWIn.img1)
    
    for channel, items in guiWIn.overlayLayersItems.items():
        imageItem, lutItem, alphaScrollBar, toolbutton = items
        
        opacities[channel] = imageItem.opacity()
        
        if opacities[channel] < 0.01:
            items_low_opacity.append(imageItem)
        elif opacities[channel] > 0.99:
            items_high_opacity.append(imageItem)
        
    
    printl(opacities, pretty=True)
    
    if not kwargs.get('do_reset_opacities', True):
        return
    
    QTimer.singleShot(
        5000, partial(
            _reset_opacities, 
            guiWIn=guiWIn,
            items_low_opacity=items_low_opacity, items_high_opacity=items_high_opacity
        )
    )

def _reset_opacities(guiWIn, items_low_opacity, items_high_opacity):
    for item in items_low_opacity:
        item.setOpacity(0.1)
    
    q_debug(guiWIn, do_reset_opacities=False)