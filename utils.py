from numpy import ones_like, zeros_like
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import time
import calendar
import math
import multiprocessing as mp
import threading

dt_dict = {
    'float':torch.float32,
    'double':torch.float64,
    'int':torch.int32
}

def iter_convert(main_miter, miter = None):
    ''' assigns values of other entries in dict/list
        to other parts of the same dict/list, supports
        nesting
        Uses _.key1.key2.key3 with a "_." to indicate
        value replace
    '''
    if miter == None:
        miter = main_miter
    if isinstance(miter, list):
        to_iter = enumerate(miter)
    elif isinstance(miter, dict):
        to_iter = miter.items()
    else:
        to_iter = []
    for key, val in to_iter:
        if isinstance(val, str):
            if val[:2] == '_.':
                keyc = val.split('.')[1:]
                sub_iter = main_miter
                for k in keyc:
                    sub_iter = sub_iter[k]
                miter[key] = sub_iter
            else:
                continue
        else:
            iter_convert(main_miter, val)
    return main_miter
    
def absmax(tensor, dim = 0):
    max_t = torch.max(tensor, dim=dim)
    min_t = torch.min(tensor, dim=dim)
    minabs_t = torch.abs(min_t)
    max_mask = max_t > minabs_t
    min_mask = 1 - max_mask
    return max_t*max_mask + min_t*min_mask

def meanmax(tensor, dim = 0):
    mean = torch.mean(tensor, dim=dim)
    max_t = torch.max(tensor, dim=dim).detach()
    k = (max_t/mean).detach()
    return k*mean
    
def weightedmax(tensor, dim = 0):
    sum_t = torch.sum(tensor, dim = dim).unsqueeze(0)
    max_t = torch.max(tensor, dim = dim).unsqueeze(0)
    weights = tensor/(sum_t + 1e-6)
    refactor = torch.sum(weights * max_t, dim=0)
    return refactor

def update_dict(base, to_up):
    '''
        It updates the dict recursively to the last leaf of
        a series of nested dictionaries
        base: dictionary to update
        to_up: dictionary with new attr
    '''
    for key, value in to_up.items():
        if isinstance(value, dict):
            if not key in base:
                base[key] = {}
            update_dict(base[key], value)
        else:
            base[key] = value

def factors(nr):
    '''Factors of number'''
    i = 2
    factors = []
    while i <= nr:
        if (nr % i) == 0:
            factors.append(i)
            nr = nr / i
        else:
            i = i + 1
    return factors

def get_receptive_field(strides=[], kernels=[]):
    '''Receptive field: approx pixels a series of conv
        layers cover in the image'''
    r_0 = 1
    for (s, k) in zip(reversed(strides), reversed(kernels)):
       r_0 = s*r_0 + (k - s)
    return r_0

def check_nans(tensor):
    '''Debug checks the nans in a tensor'''
    isn = torch.isnan(tensor)
    isf = torch.isinf(tensor)    
    index = torch.nonzero(isn)
    if isn.any() or isf.any():
        b=1
    return index
def check_big_values(tensor, value=2.0):
    '''Debug checks for big values'''
    if (tensor > value).any():
        b=1
def check_gfn(tensor):
    '''Debug checks if tensor lacks gradient function'''
    if tensor.grad_fn == None:
        b = 1
def check_version(tensor, index = 0):
    '''Debug, checks if the _version of tensor is higher than index'''
    if tensor._version > index:
        b = tensor._version
def bpp(tensor):
    '''Debug, does a backward pass to check where a grad problem
        could arise'''
    at = torch.ones_like(tensor)
    loss = torch.nn.functional.mse_loss(tensor, at)
    loss.backward(retain_graph=True)

def dynamic_average(old_value = 0, new_value = 0, n = 1):
    '''Average that only needs the last average and the number
        of values in the series
        old_value: float: old average
        new_value: float: most recent value in the serie
        n: int: lenght of the series
        ----return: int
    '''
    assert n > 0
    return (old_value*(n-1) + new_value)/n

def dynamic_average_pyt(old_value, new_value, n):
    '''Average that only needs the last average and the number
        of values in the series, for pytorch
        old_value: tensor: old average
        new_value: tensor: most recent value in the serie
        n: int: lenght of the series
        ----return: tensor
    '''
    device = old_value.device
    n = torch.tensor(n, device = device)
    return dynamic_average(old_value, new_value, n)

def create_lock(dir_path, path_lock):
    '''creates file for blocking operations (not used)'''
    if os.path.exists(dir_path):
        fd = os.open(os.path.join(dir_path, path_lock),'w')
        os.write(fd, '')
        os.close(fd)

def remove_lock(dir_path, path_lock):
    '''removes file for blocking operations (not used)'''
    path = os.path.join(dir_path, path_lock)
    if os.path.exists(path):
        os.remove(path)
        
def check_lock(dir_path, path_lock, max_cycles = 1000, cycle_time = 0.01):
    '''checks file for blocking operations (not used)'''
    max_cycles = max(1,max_cycles)
    cycle_time = max(0.01,cycle_time)

    if os.path.exists(dir_path):
        for _ in range(max_cycles):
            if os.path.exists(os.path.join(dir_path, path_lock)):
                time.sleep(cycle_time)
            else:
                return False
    else:
        return True
    return False

def torch_save_rename(to_save, dir_path, file_path, **kwargs):
    #
    if os.path.exists(dir_path):
        path, file_path = disentagle_path(dir_path, file_path)

        check_lock(dir_path, './_load_lock_' + file_path, **kwargs)
        create_lock(dir_path,'./_save_lock_' + file_path)
        torch.save(to_save, path)
        remove_lock(dir_path,'./_save_lock_' + file_path)

def torch_safe_load(dir_path, file_path, **kwargs):
    #
    if check_lock(dir_path, './_save_lock_' + file_path, **kwargs):
        return False
    else:
        path, file_path = disentagle_path(dir_path, file_path)
        if os.path.exists(path):
            create_lock(dir_path,'./_load_lock_' + file_path)
            to_return = torch.load(path)
            remove_lock(dir_path,'./_load_lock_' + file_path)
            return to_return
        
def disentagle_path(dir_path, file_path):
    #
    if len(file_path.split['/']) > 0:
        path = file_path
        file_path = file_path.split['/'][-1]
    else:
        path = os.path.join(dir_path, file_path)
    return path, file_path

def create_orphan_class_process(self, class_cn, kwargs, method_name):
    '''Creates a class object that is orphaned from the main one
        (hack)
        class_cn: constructor of obj to create
        kwargs: keyword arguments of the class
        method_name: name of method that runs the object
    '''
    mp.set_start_method('spawn')
    p = mp.Process(target=orphaned_process, args=(class_cn, kwargs, method_name))
    p.daemon = True
    p.start()
    os._exit(0)

def orphaned_process(self, class_cn, kwargs, method_name):
    '''Aux function that runs the object'''
    obj = class_cn(**kwargs)
    getattr(obj, method_name)()

def get_category(new_io):
    '''Gets the name of the terminal class
        new_io: category dict, with attrs source, type and io
        ---return: str
    '''
    return new_io['source'] + new_io['type'] + new_io['io']

def check_key(cont, key=''):
    ''' Checks if key is in cont object, covers most of the
        most common objects
    '''
    toreturn = False
    if isinstance(cont, dict):
        toreturn = True if key in cont.keys() else False
    elif isinstance(cont, (list,tuple)):
        toreturn = True if key in cont else False  
    else:
        toreturn = True if key in dir(cont) else False
    return toreturn

def check_make_folder(path):
    '''checks if path exists and if it is a folder
        if not then it creates all the folders leading
        to the path
        path: path to folder    
    '''
    if not os.path.isdir(path):
        os.makedirs(path)



def get_sample_values(values, mask = None, eps =1e-9):
    ''' gets a sample based in values given, with mask
        that multiplies the values before sampling, the
        p_i are the (log_)softmax of values
        values: tensor, n-dim
        mask: tensor that multiplies values
        eps: filters out samples that have pi less than this
        ----return:(
                sample:tensor: sample from p_i, 0-dim
                values:tensor: p_i, n-dim
            )
    '''
    #with torch.no_grad():
    #values = F.softmax(10.0*torch.tanh(values/10.0), 1)
    values = F.log_softmax(values,1)
    #values = values + torch.rand_like(values)*eps
    if not mask == None:
        values.mul_(mask)
        #mask = torch.tensor(mask, values.device).unsqueeze_(0)
    cat_dist = torch.distributions.categorical.Categorical(probs=values)
    sample = cat_dist.sample()
    if not mask == None:
        while(mask[0][sample] < eps): # make sure the category chosen is from the mask
            sample = cat_dist.sample()
    return sample, values

def get_binary_sample_values(values,  mask = None):
    ''' gets a sample based in values given, with mask
        that multiplies the values before sampling, the p_i
        are the sigmoid of values, each p_i0 is 0, then for 
        p_i1 for 1, is p_i1 = 1 - p_i0.
        values: tensor
        mask: tensor that multiplies values
        ----return:(
                sample:tensor: sample from p_i, 0-dim
                values:tensor: p_i, n-dim
            )
    '''
    if not mask == None:
        #mask = torch.tensor(mask, values.device).unsqueeze_(0)
        values.mul_(mask)
    values = torch.sigmoid(values)
    values_p = values.view(-1, values.size()[1], 1)
    values_p = torch.cat([values_p, 1-values_p], 2)
    sample = torch.distributions.categorical.Categorical(probs=values_p).sample()
    return sample, values

osct = lambda x, t : (math.sin(2*math.pi*x/t), math.cos(2*math.pi*x/t))
_time_dict_denom={
    'century':lambda x, y: osct(x, 3.1536e9),
    'year': lambda x, y: osct(x, 3.1536e7),
    'month':lambda x, y: osct(x%8.64e4 + (y[0]-1)*8.64e4, y[1]*8.64e4),
    'week':lambda x, y: osct(x%8.64e4 - y*8.64e4, 6.048e5),
    'day':lambda x, y: osct(x, 8.64e4),
    'hour':lambda x, y: osct(x, 3.6e3),
    'minute':lambda x, y: osct(x, 6e1),
    'second':lambda x, y: osct(x, 1.0),
    'ds':lambda x, y: osct(x, 0.1),
    'cs':lambda x, y: osct(x, 0.01),
    'ms':lambda x, y: osct(x, 0.001),
    'ns':lambda x, y: osct(x, 1e-6)
}
_time_dict_y = {
    x: (lambda x:None) for x in _time_dict_denom.keys()
}
_time_dict_y['week'] = lambda x : x.tm_wday
_time_dict_y['month'] = lambda x : (x.tm_mday, calendar.monthrange(x.tm_year, x.tm_mon)[1])

def time_encoding(init_time = 0.0, components = [''], flatten = True, factor = 1.0):
    '''Encoding of time difference by taking the sine and cosine of the
        difference divided by a constant of the interval of each component

        init_time: initial time
        components: intervals to take, seconds, minute, etc...
        flatten: make the list one dimensional, 1xn*2, instead of nx2
        factor: streches or contracts the time difference
        ----return encoding:list
    '''
    gmnow = time.gmtime()
    diff = (time.mktime(gmnow) - init_time) * factor
    times = []
    for tcomp in components:
        y = _time_dict_y[tcomp]
        phase_times = _time_dict_denom[tcomp](diff, y)
        if flatten:
            times.append(phase_times[0])
            times.append(phase_times[1])
        else:
            times.append(phase_times)
    return times

def get_diffuseness_io_values(diffuseness_dim, device, random = True):
    '''Returns a normalized tensor with a non-zero value
        at the indeces that category_dict refers to with
        diffuseness_keys
            category_dict: dict: categories of the terminal
            diffuseness_keys: dict: keys as the category and values as
                the index to fill
            device: torch.device:
            ----return: diff_values:tensor
    '''
    tdim = (1, diffuseness_dim)
    if random:
        diff_values = (torch.rand(
            tdim, device = device
        )*2 - 1)*1e-5 + (1/diffuseness_dim)
    else:
        diff_values = torch.full(tdim,
            1/diffuseness_dim, device=device)
    #rand number centered around 
    diff_values.requires_grad = False
    #for key, stp in category_dict.items():
    #    if key == 'io':
    #        continue
    #    indx = diffuseness_keys.index(stp)
    #    diff_values[0][indx] = 1/(len(category_dict)-2)
    return diff_values

def clean_diffuseness(diffuseness, available_stats = []):
    # TODO, adapt to new paradigm of tensor diffuseness
    for diff_comp in diffuseness:
        w_to_sort = 0
        for stat in diff_comp.keys():
            if not stat in available_stats:
                w_to_sort=+ w_to_sort
                del(diffuseness[stat])
        factor = 1 - w_to_sort
        diff_comp = {stat: diff/factor for stat,diff in diffuseness}
    return diffuseness

def get_bound_loss_fn(hard_min=0, soft_min=0, soft_max=0, hard_max=0, 
    dy=0.999, l_fac = 1.0):
    ''' Returns a function with a concave shape limited by two sigmoids
        hard_min: float: min value of left sigmoid
        soft_min: float: max value of left sigmoid
        soft_max: float: min value of right sigmoid
        hard_max: float: max value of right sigmoid
        dy: float: sharpness of sigmoids
        l_fac: float: function multiplier
        ----return: func tensor: tensor
    '''
    if soft_max == hard_max:
        max_fn = lambda values: 0
    else:
        x0 = (hard_max + soft_max)/2
        dx = hard_max - soft_max
        factor = math.log(dy/(1-dy))/(dx/2)
        #the 2 at the end because the end function is the sum of 2 sigmoids
        max_fn = lambda values: torch.sigmoid(factor*(values+x0))

    if soft_min == hard_min:
        min_fn = lambda values: 0
    else:
        x0 = (soft_min + hard_min)/2
        dx = soft_min - hard_min
        factor = math.log(dy/(1-dy))/(dx/2)
        min_fn = lambda values: torch.sigmoid(-factor*(values-x0))
    return lambda values: (l_fac*(max_fn(values) + min_fn(values))).mean()

def get_bound_loss_unbounded_fn(max_t=0.0, min_t=0.0, linear=False, mul = 1.0):
    ''' Returns a function with a concave shape limited by two unbounded
        loss functions
        max_t: float: value when function to the right starts
        min_t: float: value when function to the left starts
        linear: bool: use l1_loss, otherwise mse_loss
        mul: multiplier
        ----return: func tensor: tensor

    '''
    if linear:
        loss_fn = torch.nn.functional.l1_loss
    else:
        loss_fn = torch.nn.functional.mse_loss
    def tmp(values):
        values_abv = (values > max_t)
        max_t_T = torch.full_like(values, max_t, device = values.device)
        max_loss = loss_fn(max_t_T, values_abv*(values+max_t))
        
        values_bel = (values < min_t)
        min_t_T = torch.full_like(values, min_t, device = values.device)
        min_loss = loss_fn(min_t_T, values_bel*(values-min_t))
        return mul*(max_loss + min_loss)
    return tmp

def curiosity_value(seq):
    '''values: seq of tensors, #B, Seq, *
    '''
    dims = seq.size()
    assert len(dims) > 2
    rndval = random.randint(0,dims[1]) #uniform for now
    extra_value = seq[:, rndval]
    return extra_value

def get_normal_loss_sample(values, 
        values_loss = None, values_difference = None, loss = 1.0):
    ''' returns a randomized normal sample from values with the 
        same dimensions.
        values: tensor: mean value of distribution
        values_loss: tensor: std of distribution
            same dim as values
        values_difference: tensor: raw difference
            sets the tail it chooses
        loss: tensor(0_dim): mean loss
        ----return: normal_tensor:tensor
    '''
    miu = values#.clone()
    if not values_loss == None:
        sigma = torch.sqrt(loss * values_loss)
    else:
        sigma = loss
    normal_tensor = torch.distributions.normal.Normal(
        miu, sigma).sample()
    if not values_difference == None:
        sign = (values_difference > 0)*2 - 1
        normal_tensor = normal_tensor.sub(
            miu).abs().mul(sign).add(miu)
    return normal_tensor

def get_io_slices(slc_sizes: list, param_names: list):
    ''' function out of terminal for getting slices of parameters
    '''
    slc_io = []
    for i in range(len(slc_sizes)):
        slc_io_pre = []
        for pn in param_names:
            slc_io_pre.append(slc_sizes[i][pn])
        slc_io.append(get_slices(slc_io_pre))
    slices = {}
    for j, pn in enumerate(param_names):
        pnslc = []
        for slc_io_sub in slc_io:
            slc = slc_io_sub[j]
            pnslc.append(slc)
        slices[pn] = pnslc
    return slices

def get_slices(slc_size: list):
    '''slc_size: slice size'''
    slices = []
    s0 = 0
    for dim in slc_size:
        s1 = dim + s0
        slc = slice(s0, s1)
        slices.append(slc)
        s0 = s1
    return slices

def param_function_return(obj = None, param_names:list = None, p_func = lambda x:(x)):
    '''Aux wrapper function for methods related to parameters
        param_names:list
        p_func:func
        ----return r_v:dict
    '''
    r_v = {}
    if param_names is None or len(param_names) == 0:
        param_names = obj.param_names
    for name in param_names:
        r_v[name] = p_func(name)
    return r_v

def param_function(obj, param_names:list = None, p_func = lambda x:(x)):
    '''The same as _param_function_return without return'''
    #Function wrapper for parameter methods
    #param_names: masks the parameters that are needed
    if param_names is None or len(param_names) == 0:
        param_names = obj.param_names
    for name in param_names:
        p_func(name)

def get_memory_used(device_sett):
    ''' gets the memory used in a device
        device_sett: dict: device dict
        ----return: float: memory used
    '''
    if device_sett['type'] == 'CPU':
            avail = int(open('/proc/meminfo').read()[70:81])
            mem_used = min(1.0, 1 - avail/device_sett['max_mem'])
    elif device_sett['type'] == 'GPU':
        mem_used = int(open(
            '/sys/bus/pci/devices/%s/mem_info_vram_used' % device_sett['id']
        ).read())/device_sett['max_mem']
    return mem_used

def linear_range(values, x0=0.0, x1=1.0):
    #values: numpy or torch tensor
    m = x1-x0
    return values*m + x0

def denormalize(T, coords):
    """
    Convert coordinates in the range [-1, 1] to
    coordinates in the range [0, T] where `T` is
    the size of the image.
    """
    return (0.5 * ((coords + 1.0) * T)).long()

def exceeds(from_x, to_x, from_y, to_y, T):
    """
    Check whether the extracted patch will exceed
    the boundaries of the image of size `T`.
    """
    fx = from_x < 0
    fy = from_y < 0
    tx = to_x > T[0][0].item()
    ty = to_y > T[0][1].item()
    if fx or fy or tx or ty:
        return True
    else:
        return False

def extract_patch(x, l, size =32):
    """
    Extract a single patch for each image in the
    minibatch `x`.
    Args
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
        of images.
    - l: a 2D Tensor of shape (B, 2). in [-1.0, 1.0] form, where 0.0
        is the center of the image
    - size: a scalar defining the size of the extracted patch.
    ----Returns patch: a 4D Tensor of shape (B, size, size, C)
    """
    B, _, H, W = x.shape
    device = x.device

    # denormalize coords of patch center
    T = torch.zeros(B,2, device=device)
    T.index_fill_(1, torch.tensor([0], device=device), W)
    T.index_fill_(1, torch.tensor([1], device=device), H)

    coords = denormalize(T,l)

    # compute top left corner of patch
    patch_x = coords[:, 0] - (size // 2)
    patch_y = coords[:, 1] - (size // 2)

    # loop through mini-batch and extract
    patch = []
    for i in range(B):
        coord_i = coords[i]
        im = x[i].unsqueeze(dim=0)
        #T = im.shape[-1]

        # compute slice indices
        from_x, to_x = patch_x[i], patch_x[i] + size
        from_y, to_y = patch_y[i], patch_y[i] + size

        # cast to ints
        from_x, to_x = from_x.item(), to_x.item()
        from_y, to_y = from_y.item(), to_y.item()

        # pad tensor in case exceeds
        if exceeds(from_x, to_x, from_y, to_y, T):
            z0 = torch.tensor([0], device=device)
            add_pad_x = torch.max(z0, coord_i[0] - W) - torch.min(z0, coord_i[0])
            add_pad_y = torch.max(z0, coord_i[1] - H) - torch.min(z0, coord_i[1])
            pad_x = add_pad_x + size//2+1
            pad_y = add_pad_y + size//2+1
            pad_dims = (
                pad_x, pad_x,
                pad_y, pad_y,
                0, 0,
                0, 0,
            )
            im = F.pad(im, pad_dims, "constant", -1.0)

            # add correction factor
            from_x += pad_x
            to_x += pad_x
            from_y += pad_y
            to_y += pad_y

        # and finally extract
        patch.append(im[:, :, from_y:to_y, from_x:to_x])
    # concatenate into a single tensor
    patch = torch.cat(patch)

    return patch

def foveate(x, l, size=32, patches = []):
    """
    Extract `k` square patches of size `g`, centered
    at location `l`. The initial patch is a square of
    size `g`, and each subsequent patch is a square
    whose side is `s` times the size of the previous
    patch.

    The `k` patches are finally resized to (g, g) and
    concatenated into a tensor of shape (B, k, g, g, C).
    """
    phi = []
    # extract k patches of increasing size
    for i in range(len(patches)):
        size_p = int(patches[i] * size)
        phi.append(extract_patch(x, l, size_p))
    # resize the patches to squares of size g
    for i in range(1, len(phi)):
        k = phi[i].shape[-1] // size
        phi[i] = F.avg_pool2d(phi[i], k)
    phi = torch.stack(phi, dim = 1)
    # concatenate into a single tensor and flatten
    #phi = torch.cat(phi, 1)
    #phi = phi.view(phi.shape[0], -1)
    return phi
affn_limit_presets ={
    'intrinsic':{
        'rotation': [-0.05, 0.05],
        'scale': [0.9, 1.1],
        'location':[-0.05, 0.05],
        'skew':[-0.025,0.025]
    },
    'extrinsic':{
        'rotation': [-0.5, 0.5],
        #'scale': [-2.0,-0.5, 0.5, 2.0], 
        'scale': [0.5, 2.0], 
        'location':[-0.1, 1.1],
    },
    'none':{
    }
}
def limit_fn(x, limits):
    ''' limits the value of x to range or ranges defined in limits
    '''
    x = torch.nn.Tanh()(x)*max(limits)
    if len(limits) == 4:
        x_p = x.clone() > 0
        x_n = x.clone() < 0
        x_out_p = torch.clamp(x.clone() * x_p, limits[0], limits[1])
        x_out_n = torch.clamp(x.clone() * x_n, limits[2], limits[3])
        x_out = x_out_p + x_out_n
    elif len(limits) == 2:
        x_out = torch.clamp(x, limits[0], limits[1])
    return x_out

def transform2affine_pyt(rotation:torch.Tensor = None, scale:torch.Tensor = None, 
        location:torch.Tensor=None, skew:torch.Tensor=None, limits='none'):
    ''' takes primitive transforms and coverts them into an affine transform
    '''

    batch_size = rotation.size()[0]
    transforms = {
        'rotation': rotation,
        'scale': scale,
        'location': location,
        'skew': skew,
    }
    transform_values = {}
    for comp, tvalue in transforms.items():
        if tvalue == None:
            continue
        elif tvalue in affn_limit_presets:
            transform_values[comp] = limit_fn(
                tvalue, affn_limit_presets[limits][comp])
        else:
            transform_values[comp] = tvalue
    affn = torch.zeros(batch_size, 2, 3, device=rotation.device)
    affn[:, 0, 0] += 1.0
    affn[:, 1, 1] += 1.0
    affn = nn.Parameter(affn)
        
    if not rotation is None:
        rot = transform_values['rotation']*math.pi
        rot_sin = torch.sin(rot.clone())
        rot_cos = torch.cos(rot).unsqueeze(-1).unsqueeze(-1)
        affn = affn * rot_cos
        affn[:, 0, 1] = affn[:, 0, 1] + -rot_sin
        affn[:, 1, 0] = affn[:, 0, 1] + rot_sin
    if not skew is None:
        skew = transform_values['skew']
        affn[:, 0, 1] = affn[:, 0, 1] + skew[:, 0]
        affn[:, 1, 0] = affn[:, 1, 0] + skew[:, 1]
    if not scale is None:
        scale = transform_values['scale']
        affn = affn * scale.unsqueeze(2)
    if not location is None:
        loc = transform_values['location']
        affn[:, :, 2] = affn[:, :, 2] + loc
    return affn

def safe_open(file_path, waittime =  0.01):
    not_file = True
    i = 0
    while(not_file and i < 100):
        try:
            f = open(file_path, 'a+')
            not_file = False
        except:
            time.sleep(waittime)
            i = i + 1
    if not_file:
        f = None
    return f

def safe_open_text_file(file_path, check='none', waittime = 0.005):
    not_cond = True
    if check == 'empty':
        cond_fn = lambda x : len(x) == 0
    if check == 'content':
        cond_fn = lambda x : len(x) > 0
    else:
        cond_fn = lambda x : True
    i = 0
    while(not_cond and i < 100):
        f = safe_open(file_path, waittime)
        if f is None:
            i = i + 1
            time.sleep(waittime)
        text = f.read()
        if cond_fn(text):
            not_cond = False
        else:
            f.close()
            time.sleep(waittime)
            i = i + 1
    if not_cond:
        f = None
        text = ''
    return f, text

def get_routing_masks(settings, cell_types='all',types='all', sources='all', 
    devices = None, strict = False):
    ''' gets the routing masks that are later stored in each cell in
        the chains that the terminal spawns, used for directing the 
        routing, if 'all' in each kwarg then it sets true in all items
        of the vector mask
        cell_types: str: cell, terminal, kill types
    '''

    conn_masks = settings['conn_masks']
    all_types = settings['all_types']
    all_sources = settings['all_sources']
    acell_types = settings['cell_types']
    dev_masks = {}
    if not isinstance(devices, (list, tuple)):
        devices = [devices]
    for dev in devices:
        r_masks = {
            i:conn.detach().clone().to(device=dev)
            for i, conn in conn_masks.items()
        }
        if cell_types == 'all':
            cell_types = acell_types
        if types == 'all':
            types = all_types
        if sources == 'all':
            sources = all_sources
        mask_sett = [cell_types, types, sources]
        mask_all = [acell_types, all_types, all_sources]

        for idx in range(len(mask_sett)):
            msk_items = mask_sett[idx]
            for i,itm in enumerate(msk_items):
                if itm in mask_all[idx]:
                    mid = mask_all.index(itm)
                    r_masks[idx][mid] = 1.0
                elif strict:
                    raise Exception('Incorrect mask element '+ itm)
        dev_masks[dev] = r_masks
    return dev_masks

def entropy_loss(values):
    ''' entropy_loss
    '''
    log_p = torch.log_softmax(values, dim = 1)
    p = torch.softmax(values, dim=1)
    return - (log_p*p).mean() #maximize

def uniform_policy_loss(policy, actions = None):
    ''' classic uniform policy
    '''
    if not actions:
        loss = policy
    else:
        loss = F.cross_entropy(policy, actions)
    return loss

def gaussstdvar_loss(policy):
    pols = policy.size()[-1]
    policy_std = policy[:,pols//2:]
    return - policy_std #maximize

def gaussian_policy_loss(policy, actions):
    '''gaussian policy with stadard deviation and mean'''
    pols = policy.size()[-1]
    policy_mean = policy[:,:pols//2]
    policy_std = policy[:,pols//2:]
    loss = F.gaussian_nll_loss(policy_mean, actions, policy_std)
    return loss

class CellLockWrap(object):
    def __init__(self, cell, local_source, keylock, wait_time):
        self.cell = cell
        self.local_source = local_source
        self.keylock = keylock
        self.wait_time = wait_time
    def __enter__(self):
        if self.cell.check_lock(self.local_source, self.keylock):
            time.sleep(self.wait_time)
        self.cell.lock(self.local_source, self.keylock)
    def __exit__(self, exception_type, exception_value, traceback):
        self.cell.unlock(self.local_source, self.keylock)


