'''
    script for offline operations, like reseting the cells,
    mutate the cells, or expand the cells, or set cells that are not
    present
    --preset_name: base preset to use in presets
    --subpreset: additional preset replacing only the keys/values
        given
    --mode: 
        create: creates non-existing cells that should exist
        resets: resets all cells
        expand: expands the models of existing cells according to
            the new settings, also creates the new cells
        load: only loads the models, this is a debug option
        mutate: mutates the model's parameters with the best cells
            closest in diffuseness

'''
if __name__ == "__main__":
    #import torch
    import classes
    import presets
    #from multiprocessing import Manager, Process
    import argparse
    import presets.subpresets as subpresets
    from utils import update_dict
    from settings import post_process_settings

    parser = argparse.ArgumentParser(
        description='Universal Learner'
    )
    parser.add_argument('--preset_name', default='default', type = str)
    parser.add_argument('--subpreset', default='test', type = str)
    parser.add_argument('--mode', default='create', type = str, 
        choices=['create', 'reset', 'expand', 'load', 'mutate'])
    args = parser.parse_args()
    init_settings = getattr(presets, args.preset_name).settings
    subsett = getattr(subpresets, args.subpreset)
    update_dict(init_settings, subsett)
    acc_devices = {
        'cpu': init_settings['memory']['devices']['cpu']
    }
    settings = post_process_settings(
        init_settings, acc_devices=acc_devices, wake_phase='all'
    )
    if args.mode == 'reset':
        c1 = input('Are you sure you want to reset all? [y/n]')
        c2 = input('Sure? [y/n]')
        print('Suffer :DDDDD')
        if not c1.lower() == 'y' and not c2.lower() == 'y':
            print('Exit')
            exit()

    obj_terms, _ = classes.get_terminals(settings)
    '''for io in obj_terms:
        for obj in obj_terms[io]:
            print(obj.category, obj.param_names, obj.parameters_settings)
    '''
    Hmemory = classes.CellHierarchicalMemoryManagerOffline(
        obj_terms, main_device = 'cpu', acc_devices=acc_devices,
        mode = args.mode, settings=settings
    )
