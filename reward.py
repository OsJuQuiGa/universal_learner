'''
    parent to be run in parent session, it includes:
    1)  reward ui, it makes the user enter the sources to value
        and the reward, negative or positive between 0 and 1, opened in
        master session
'''
import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk
from utils import safe_open_text_file
def set_reward_UI(settings):
    #from os import system

    root = tk.Tk()
    default_font = tkFont.nametofont("TkDefaultFont")
    default_font.configure(size=14)
    root.title('Rewards Panel')
    window = ttk.Frame(master=root)
    window.grid()
    real_sources = []
    for i, (term_name, types) in enumerate(settings['types_available'].items()):
        #checks if there's a real terminal in each source
        if not 'Real' in types:
            continue
        real_sources.append(term_name)
    rewards_values = {}
    rewards_to_send = {}
    rewards_nll_bttns = {}
    labels_frame = ttk.Frame(master = window)
    rewards_frame = ttk.Frame(master = window)
    submit_frame = ttk.Frame(master = window)
    send_frame = ttk.Frame(master = window)
    past_frame = ttk.Frame(master=window)
    labels_frame.grid(row=0, column=0)
    rewards_frame.grid(row=0, column=1)
    submit_frame.grid(row=1, column=1)
    send_frame.grid(row=1, column=0)
    past_frame.grid(row=0, column=2, rowspan=2)

    def upt_rwd_slc(rwdlbl, rwdsnt):
        def tmp(val):
            val = float(val)
            spacer = ' ' if val >= 0.0 else ''
            rwdlbl.set(spacer+"%.3f" % val)
            rwdsnt.set(True)
        return tmp
    def upt_nll_rwd_bttn(rwdsnt, slider):
        def tmp():
            rwdsnt.set(False)
            slider.set(0.0)
        return tmp
        
    for i, src in enumerate(real_sources):
        rwdsrc = tk.StringVar()
        rwdsnt = tk.BooleanVar()
        rwdlbl = tk.StringVar()
        #rwdsrc.set(0)
        rwdsrc.set(False)
        rwdlbl.set('0.00')
        rewards_values[src] = rwdsrc
        rewards_to_send[src] = rwdsnt
        
        label = ttk.Label(master=labels_frame, text=src)
        label.grid(row=i, column=0)
        rwdlabel = ttk.Label(master=rewards_frame,
            textvariable=rwdlbl, width=8)
        rwdlabel.grid(row=i, column=2)
        cmd = upt_rwd_slc(rwdlbl, rwdsnt)

        slider = ttk.Scale(master=rewards_frame, orient='horizontal',
            length=200, from_=-1.0, to=1.0, variable=rwdsrc,
            command= cmd)
        slider.grid(row=i, column=1)
        cmd = upt_nll_rwd_bttn(rwdsnt, slider)

        slider.set(0.000)
        rwdsrc.set(0.000)
        
        nllbutton = ttk.Button(master=rewards_frame, text='null',
            command=cmd)
        nllbutton.grid(row=i, column=3)
        rewards_nll_bttns[src] = nllbutton

    def send_text(*args):
        text = text_wgt.get('1.0', 'end')
        if len(text) == 0:
            return
        f, _ = safe_open_text_file('./text_message', check='empty')
        f.write(text)
        f.close()
        text_wgt.delete('1.0', 'end')

    def submit(*args):
        lines = []
        for src, rwd in rewards_to_send.items():
            if rwd.get():
                arwd = rewards_values[src].get()
                lines.append(':'.join([src, arwd]))
        text = ';'.join(lines)
        f, _ = safe_open_text_file('./reward', check='empty')
        f.write(text)
        f.close()
        past_submit.delete('1.0', 'end')
        past_submit.insert('1.0', '\n'.join(lines))
        reset(*args)

    def reset(*args):
        for buttn in rewards_nll_bttns.values():
            buttn.invoke()

    text_wgt = tk.Text(master=send_frame)
    text_wgt.grid()
    submit_button = ttk.Button(
        master=submit_frame, text='Submit', command=submit)
    submit_button.grid(row=0, column=2)
    reset_button = ttk.Button(
        master=submit_frame, text='Reset', command=reset)
    reset_button.grid(row=0, column=1)
    send_button = ttk.Button(
        master=submit_frame, text='Send', command=send_text)
    send_button.grid(row=0, column=0)

    past_submit = tk.Text(master=past_frame)
    past_submit.grid(row=0, column=0)

    tk.mainloop()
    
if __name__ == "__main__":
    pass
