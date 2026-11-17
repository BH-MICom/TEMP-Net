"""Measured inter-segment cooling estimate and bounded excess-temperature removal."""
import numpy as np

def estimate_cooling(previous_end, next_baseline, initial_baseline, tumor, foreground, rise_threshold_C=2.):
    valid=(np.asarray(tumor,dtype=bool)&np.asarray(foreground,dtype=bool)
           &np.isfinite(previous_end)&np.isfinite(next_baseline)&np.isfinite(initial_baseline)
           &(previous_end>0)&(next_baseline>0)&(initial_baseline>0))
    # A fixed previous-heating ROI: never recenter it at the new probe position.
    heated=valid&((previous_end-initial_baseline)>rise_threshold_C)
    if not heated.any():
        return {'status':'no_measurable_previous_heated_TC_pixels','pixels':0,'raw_mean_drop_C':None,
                'mean_previous_end_C':None,'mean_next_baseline_C':None,'median_drop_C':None}
    drop=np.asarray(previous_end,np.float64)[heated]-np.asarray(next_baseline,np.float64)[heated]
    return {'status':'estimated','pixels':int(heated.sum()),'raw_mean_drop_C':float(drop.mean()),
            'mean_previous_end_C':float(np.mean(previous_end[heated])),
            'mean_next_baseline_C':float(np.mean(next_baseline[heated])),
            'median_drop_C':float(np.median(drop))}

def combine_plane_estimates(estimates):
    values=[p['raw_mean_drop_C'] for p in estimates if p['raw_mean_drop_C'] is not None]
    mean=float(np.mean(values)) if values else 0.
    return max(0.,mean),mean

def bounded_cooling(temperature, initial_temperature, brain, delta_C):
    if not np.isfinite(delta_C) or delta_C<0:raise ValueError('Cooling delta must be finite and nonnegative')
    result=np.asarray(temperature).copy()
    brain=np.asarray(brain,dtype=bool)
    excess=np.maximum(result[brain]-np.asarray(initial_temperature)[brain],0.)
    result[brain]-=np.minimum(delta_C,excess)
    return result
