'''
    main script for Universal slave, the infite loop is done externally
    in bash
    args:
        --preset_name: base preset to use in presets
        --subpreset: additional preset replacing only the keys/values
            given
        --phasehours: range of hours in which each phase runs, it could be
            int,int -> hour for start wake, hour for start dream
            int -> hours for each interval, starting for wake
            'wake' or 'dream' -> force a phase
        --smooth_phasing: gradually changes to each phase in the hour before changing phase,
            memory intensive becase it loads all terminals
        --accdevice: acceleration device to use
        --ndevices: if cuda, use the devices in [x,y,z] order, separated by commas
        --prestart: allow the program to wait for user input to start and set
            a timer for starting the program
'''
import time
import argparse

def argparser():
    parser = argparse.ArgumentParser(
        description='Universal Learner'
    )
    parser.add_argument('--preset_name', default='default', type = str)
    parser.add_argument('--subpreset', default='none', type=str)
    parser.add_argument('--phasehours', default='23', type = str)
    parser.add_argument('--smooth_phasing', default=0, type = int)
    parser.add_argument('--accdevice', default='cuda', type=str, 
        choices=['cuda', 'cpu'])
    parser.add_argument('--ndevices', default='0', type=str)
    parser.add_argument('--prestart', default=0, type = int)
    parser.add_argument('--init_behaviour', default='set_and_load', type=str)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--rank', default='slave')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    from universal import UniversalSlave, UniversalMaster

    args = argparser()
    if args.prestart > 0:
        input('Press Enter to start')
        print('Universal will start in...')
        for i in reversed(range(args.prestart)):
            time.sleep(1.0)
            print(str(i) + '...')
    if args.rank == 'slave':
        universal = UniversalSlave(args)
    elif args.rank == 'master':
        universal = UniversalMaster(args)
    universal.continual_run()





                


