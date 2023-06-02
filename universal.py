import time
import os
import torch
import utils
import presets
import random
import pickle
import classes
import presets.subpresets as subpresets
from deepdiff import DeepDiff
from settings import post_process_settings
from utils import dynamic_average, update_dict
from work import Cycle

class Universal():
    def __init__(self, args):
        self.args = args
        #---------INIT SETTINGS----------------
        init_settings = getattr(presets, args.preset_name).settings
        subsett = getattr(subpresets, args.subpreset)
        update_dict(init_settings, subsett) #overwrittes only the end values of dicts, not dicts themselves
        self.init_settings = init_settings
        self.check_settings(init_settings) #checks if there're changes

        self.set_devices() #sets the devices and the order of the devices
        #---------WAKE TABLE----------------
        self.set_wake_table()
        #--------------PRELOOP---------------------
        self.stats = {
            'big_loss_avg':1.0,
            'big_loss_max':self.init_settings['day_big_loss_max'],
            'big_loss_change_avg': 0.1,
            'init_cycles':1,
            'global_cycles': 0,
        }
        self.update_day = False
        self.p_t = time.time()
        self.avg_time = 1.0
        self.phase = {}
        self.set_phase(time.localtime())
        self.prev_wph = self.phase['wph']
        
        self.settings = post_process_settings(
            self.init_settings, acc_devices=self.acc_devices,
            wake_phase= self.phase['wph']
        )
        self.slockfp = './external/slave_lock'
        self.mlockfp = './external/master_lock'
        print('Phase Change to ' + self.phase['wph'])


    def set_devices(self,):
        '''
            sets the devices to use in pytorch semantics 'cuda:0' for example
            uses args.accdevice to use either cpu or cuda, and args.ndevices
            to set the order of cuda devices, ex: 1,0 makes cuda:1 the maindevice
            and cuda:0 the device to store other terminal/cell objects
        '''
        acc_devices = {}
        if 'cpu' == self.args.accdevice:
            maindevice = 'cpu'
            dev_indx = []
        else:
            dev_indx = [int(x) for x in self.args.ndevices.split(',')]
            dev_order = [None]*len(dev_indx)
            if len(dev_indx) == 0:
                #if not dev index is given it searches for the smallest one
                min_indx = 0
            else:
                #otherwise uses this value to use it as the maindevice
                min_indx = dev_indx[0]
            
            for name, dev in self.init_settings['memory']['devices'].items():
                assert dev['use'] #makes sure the device is available
                if len(dev_indx) == 0:
                    min_indx = max(min(min_indx, dev['idx']),0)
                for dindx in dev_indx:
                    if dev['idx'] == dindx:
                        acc_devices[name] = dev
                        dev_order[dev_indx.index(dindx)] = name
            maindevice = self.args.accdevice+':'+ str(min_indx) 
        if len(dev_indx) == 0:
            dev_order = [maindevice]
            acc_devices = {maindevice:self.init_settings[
                'memory']['devices'][maindevice]}
        self.dev_order = dev_order #order of devices
        self.acc_devices = acc_devices #acceleration devices
        self.maindevice = maindevice

    def check_settings(self, settings):
        '''
            settings: dict object
            checks if the settings have been changed and also
            makes a backup of the current settings
        '''
        old_sn = self.args.preset_name + '_old'
        old_sn_path = os.path.join('./presets', old_sn + '.plk')
        f = open(old_sn_path, 'wb+')
        if os.path.exists(old_sn_path) and os.path.getsize(old_sn_path)>0:
            old_settings = pickle.load(f)
            diff = DeepDiff(settings, old_settings)
        else:
            pickle.dump(settings, f)
            diff = {}
        
        if len(diff) > 0:
            print('preset file '+ self.args.preset_name+' modified')
            yn = input('Continue? y/n')
            if yn.lower() == 'y':
                pickle.dump(settings, f)
            else:
                print('Not running...')
                exit()

    def set_phase(self, localtime):
        '''
            localtime : time.localtime object
            sets the phase of each run, it uses the wake table to
            get the phase to use, also adjust the source of the Dream
            terminal and the Critic when dreaming, also changes the lr
        '''
        phase = self.phase
        hour = localtime.tm_hour + self.init_settings['utc_hour'] 
        #utc_hour adjust for current geographic time
        minute = localtime.tm_min
        wph = self.wake_table[hour]
        stats = self.stats
        phase['wph'] = wph
        if wph == 'wake': 
            phase['wake'] = True
            phase['focus_ratio'] = 1.0
        elif wph == 'dream': 
            phase['wake'] = False
            phase['focus_ratio'] = 0.0
        else:
            p = minute/60
            sam = random.random()
            if wph=='dream2wake': 
                phase['wake'] = p > sam
            else: 
                phase['wake'] = 1-p > sam
            phase['focus_ratio'] = p
        if not phase['wake']:
            if stats['big_loss_avg'] > stats['big_loss_max']:
                phase['dream_init'] = 'critic'
            elif stats['big_loss_avg'] > stats['big_loss_max']/2:
                phase['dream_init'] = 'virtual'
            else:
                phase['dream_init'] = 'self'
            sam = random.random()
            if sam > 0.006:
                phase['dream_enc'] = 'dream'
            elif sam > 0.001:
                phase['dream_enc'] = 'noise'
            else:
                phase['dream_enc'] = 'old'
        #phase['routing_wgt'] = 1/(1+(stats['big_loss_change_avg']-0.25))
        #phase['routing_wgt'] = 1.0
        phase['entropy'] = 1/(stats['big_loss_avg']+0.1) 
        self.phase = phase

    def set_wake_table(self):
        '''
            initializes the wake table, it has 2 modes, 2 ints where
            the first is the hour to get awakened and the other the hour
            to start dreaming, one int for the hours in each wake/dream phase,
            one string with either 'wake' or 'dream' to force the phase, this
            wake table is then is set to the object
        '''
        phhours = self.args.phasehours.split('-')
        if len(phhours) == 1:
            if phhours[0] == 'wake' or phhours[0] == 'dream':
                wake_table = [phhours[0]]*24
            else:
                phase = 'wake'
                i = 0
                wake_table = []
                phhours = int(phhours[0])
                for x in range(24):
                    if i > phhours-1:
                        if phase == 'wake':
                            phase = 'dream'
                            if self.args.smooth_phasing:
                                wake_table.append('wake2dream')
                        elif phase == 'dream':
                            phase = 'wake'
                            if self.args.smooth_phasing:
                                wake_table.append('dream2wake')
                        i = 0
                        if self.args.smooth_phasing:
                            continue
                    wake_table.append(phase)
                    i = i + 1
            self.wake_table = wake_table
        elif len(phhours) == 2:
            day, night = [int(x) for x in phhours]
            wake_table = ['wake' if x>day and x<night else 'dream' for x in range(24)]
            if self.args.smooth_phasing:
                wake_table[day] = 'dream2wake'
                wake_table[night] = 'wake2dream'
        
        self.wake_table = wake_table

class UniversalMaster(Universal):
    def __init__(self, args):
        super().__init__(args)
        self.Hmemory = classes.CellHierarchicalMemoryManagerMaster(
            self.maindevice, phase = self.phase,
            acc_devices = self.acc_devices, dev_order = self.dev_order,
            init_behaviour='parent', settings=self.settings
        )
        self.remove_lock()

    def remove_lock(self,):
        if os.path.exists(self.mlockfp):
            os.remove('./external/master_lock')

    def set_lock(self,):
        '''sets and removes lock for master/slave'''
        if os.path.exists(self.slockfp):
            os.remove(self.slockfp)
        f = open(self.mlockfp, 'x')
        f.write('')
        f.close()

    def wait_for_slave(self,):
        '''waits for slave'''
        while(os.path.exists('./external/master_lock')):
            time.sleep(0.01)

    def continual_run(self,):
        '''continual run, in slave this wait/lock is
        in the cyclegraph class run method'''
        cond = True
        while(cond):
            cond = self.run_cycle()
    
    def run_cycle(self,):
        self.wait_for_slave()
        self.prev_wph = self.phase['wph'] #keeps track of previous phase for phase change
        localtime = time.localtime()
        self.set_phase(localtime)
        if self.phase['wph'] == self.prev_wph:
            for in_terminal in self.Hmemory.imitate_input_terminals:
                #traverses only the values that are shared later
                _ = in_terminal.forward_fn(
                    self.Hmemory, self.phase)
                in_terminal.set_parameter_values(
                    in_terminal.parameters, extra = 'master'
                )
                _ = in_terminal.save_parameter_values(extra = 'master')
            if not self.args.debug:
                self.set_lock()
            return True
        else:
            return False

class UniversalSlave(Universal):
    def __init__(self, args):
        '''
            Inits the Slave
            1) gets the base settings, then the subsettings in args
                that updates the base settings, and post-process these settings
            2) gets the memory manager with the settings in 1) and the devices to use
                in self.set_devices()
            3) sets a bunch of vzriables like the stats, the time, phase, antilearn_lr
                etc.
        '''
        super().__init__(args)
        self.Hmemory = classes.CellHierarchicalMemoryManagerSlave(
            self.maindevice, phase = self.phase,
            acc_devices = self.acc_devices, dev_order = self.dev_order,
            init_behaviour= args.init_behaviour, settings=self.settings
        )
        self.gs_path = os.path.join(
            self.Hmemory.experiments_folder,
            'global_statistics.pth'
        ) #overwrite stored statistics
        if os.path.exists(self.gs_path):
            self.stats = torch.load(self.gs_path, 
                map_location=self.maindevice)

        self.init_cycles = self.Hmemory.cycles
        self.day_big_loss_max = self.init_settings['day_big_loss_max']
        self.phase['phase_change_lr'] = 0.0
        self.phase['antilearn_l'] = 1.0
        self.phase['antilearn_cum'] = 0
        self.to_min_lr = 1.0 if self.phase['wake'] else 0.25
        self.out_lr = -1.0 #out_lr is a lr that is clipped to 0.0 and 1.0
        self.cycle = Cycle(
            self.Hmemory, self.args.inspect, 
            self.settings
        )

    def unlock_all(self,):
        for p in [self.slockfp, self.mlockfp]:
            if os.path.exists(p):
                os.remove(p)

    def continual_run(self,):
        '''
            Infinite loop that runs the run cycle method, breaks when there's
            a phase change, saving btw the state of all cells and terminals
        '''
        cond = True
        if self.settings['batch_imitation'] and self.phase['wake']:
            self.cycle.set_lock() #init lock for slave
            print('Initial wait for Parent...')
        while(cond):
            cond = self.run_cycle()
            if self.settings['debug_mode']:
                cond = self.Hmemory.cycles % 8 == 7
        self.unlock_all()
        print('Finnishing ' + self.prev_wph + ' phase')
        #self.Hmemory.cross_terminals(self.prev_wph)
        self.Hmemory.update_states_offline()
        torch.save(self.stats, self.gs_path) #save global stats

    def run_cycle(self,):
        '''
            method that executes the cycle, phase, chaingraph, and checkpoint save
            1) sets the phase, lr, use past or not, entropy etc, then changes the direction
                of the lr if the antilearn mechanism is activated
            2) initializes the Cycle clas and runs it, the results are processed for self.stats
            4) Prints the results and other statistics like lr and phase
        '''
         #Time in 0 UTC
        self.prev_wph = self.phase['wph'] #keeps track of previous phase for phase change
        localtime = time.localtime()
        self.set_phase(localtime)
        #Chain cycle
        #self.set_antilearn(self.phase) don't set
        if self.phase['wph'] == self.prev_wph:
            stats = self.stats #updates self.phase, lr, and inits the phase change
            big_loss, big_loss_change = self.cycle.run(self.phase)
            self.stats['global_cycles'] += 1
            offline_save = self.Hmemory.update_states() #saves checkpoints every x000th time
            if offline_save: 
                self.out_lr = 0.0 #resets lr when doing offline save
                #because the cycle is disrupted
                torch.save(self.stats, './global_statistics.pth')

            #------------ UPDATE THE GLOBAL STATISTICS --------------
            stats['big_loss_avg'] = dynamic_average( #keeps average loss for phase update
                stats['big_loss_avg'], big_loss, 
                100
            )
            stats['big_loss_change_avg'] = dynamic_average(
                stats['big_loss_change_avg'], big_loss_change, 
                100
            )
            self.day_big_loss_max = max(
                big_loss.cpu().item(), self.day_big_loss_max
            ) #keeps the maximum loss

            if not localtime.tm_hour == 1 + self.settings['utc_hour']:
                self.update_day = False
            elif not self.update_day + self.settings['utc_hour']:
                self.update_day = True
                stats['big_loss_max'] = self.day_big_loss_max
                self.Hmemory.cycles = self.Hmemory.cycles % self.Hmemory.save_freq

            #----------- TIME DECAY -----------------
            now_t = time.time()
            prev_t = self.p_t
            self.p_t = now_t
            #------------- DYN LR -------------------
            self.out_lr += 1/self.settings['phase_change_t']
            self.phase['phase_change_lr'] = max(0.0,
                min(self.to_min_lr, self.out_lr)
            )
            # slowly raises the lr when phase is changed or when there's a checkpoint 
            # save and clips
            self.avg_time = utils.dynamic_average(self.avg_time, now_t-prev_t, 10)
            sf = self.Hmemory.save_freq
            if self.Hmemory.cycles % 10 == 0:
                print(' _ '.join([
                        'cycle: [' + str(self.Hmemory.cycles % sf) + '/' + str(sf) + ']',
                        'phase: ' + self.phase['wph'],
                        'loss: ' + '%.4f' % big_loss.cpu().item(),
                        'lr: ' + '%.4f' % self.phase['phase_change_lr'],
                        'time: ' + '%.2f' % self.avg_time,
                        'learn: ' + '%.2f' % self.phase[
                            'antilearn_l'] +'|'+ str(self.phase['antilearn_cum']),
                    ])
                )
            keep_going = True
        else:
            keep_going = False
        return keep_going
        
    def set_antilearn(self, phase):
        '''
            sets or not the negative lr
            it accumulates cycles that have big losses above a threshold,
            if surpassed then it will switch the direction of the lr for
            some cycles.
        '''
        if phase['antilearn_l'] < 0.0:
            phase['antilearn_cum'] -= 1
            if phase['antilearn_cum'] < 0:
                phase['antilearn_l'] = 1.0
        else:
            anti_set = self.settings['antilearn']
            if self.stats['big_loss_avg'] < anti_set['threshold_min']:
                phase['antilearn_cum'] +=1
            elif self.stats['big_loss_avg'] > anti_set['threshold_max']:
                phase['antilearn_cum'] -= 1
                phase['antilearn_cum'] = max(
                    -anti_set['activation'], phase['antilearn_cum'])
            if phase['antilearn_cum'] > anti_set['activation']:
                phase['antilearn_l'] = -0.5
                phase['antilearn_cum'] = anti_set['duration']