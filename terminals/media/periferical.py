import time
import pyautogui
def write_key_sequence(sequence, cmds_fn, time_item = 0.01):
    '''
        sequence: List of keystrokes
        (key to push index, command)

    '''
    for key_comp in sequence:
        key, command = key_comp
        if key == 'SOS': pass
        elif key == 'EOS': return
        elif key == 'null': time.sleep(time_item)
        else:
            cmdf = cmds_fn(command)
            cmdf(key)
            time.sleep(time_item)

def play_mouse_sequence(sequence, cmds_fn, time_item = 0.01):
    for key_comp in sequence:
        key, command, l, scroll = key_comp
        if key == 'SOS': 
            continue
        elif key == 'EOS': 
            return
        elif key == 'null':
            pass
        else:
            cmdf = cmds_fn(command)
            cmdf(key)
        pyautogui.scroll(scroll)
        pyautogui.moveTo(l, time_item) #blocking function

        #time.sleep(time_item)