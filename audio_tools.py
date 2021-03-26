import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def stft( input_sound, dft_size, hop_size, zero_pad, window):
    """
    Computes STFT of a given 1D audio 
    @params:
        input_sound: 1D (num samples, ) numpy array
        dft_size: int ie 512
        hop_size: int ie 128
        zero_pad: zero_padding each frame's DFT
        window: windowing function, ie np.hanning(512)
    @returns:
        spectrogram of (nbins, num frames) Complex type
    """
    nframes = 1 + int((len(input_sound)-dft_size)/hop_size)
    nfreqbins = int(dft_size/2)+1
    # make sure it is complex so we won't lose infomration for DFTs
    f_t = np.zeros((nframes,nfreqbins), dtype=np.complex)
   
    
    for i in range(nframes):
        st = i*hop_size 
        ed = i*hop_size + dft_size
        
        if(ed >len(input_sound)):
            last_frame = np.zeros(dft_size)
            last_frame[0:len(input_sound[i*hop_size:])] = input_sound[i*hop_size:]
            X = np.fft.rfft(last_frame*window, n = dft_size+zero_pad)
        else:
        
            X = np.fft.rfft(input_sound[st:ed]*window, n = dft_size+zero_pad)
        f_t[i] = X        
    # Return a complex-valued spectrogram (frequencies x time)
    return f_t.T

def istft( stft_output, dft_size, hop_size, zero_pad, window):

    #test for dim, if 1d call STFT instead and return
    if len(stft_output.shape) <2:
        return stft( stft_output, dft_size, hop_size, zero_pad, window)
    
    nframes = (stft_output.shape)[1]
    nfreqbins = (stft_output.shape)[0]
    
    x = np.zeros((nframes-1)*hop_size+dft_size)
   
    
    # so now (nframes, nfreqbins) 
    stft_output = stft_output.T
    for i in range(nframes):
        st = i*hop_size 
        ed = i*hop_size + dft_size+zero_pad
        x[st:ed] += window*(np.fft.irfft(stft_output[i]))
        
    
    # Return reconstructed waveform
    return x



def to_float(audio):
    """
    Converts np.int16 (2^16)/2 = 32768 to float32. note this is signed therefore 32758 is the biggest number we can represent 
    @params:
        audio: np array of (number of samepls,), yes this is 1D array
    """
    return audio.astype(np.float32, order='C') / 32768.0


def plot_spec(spec,x,sr ,title):
    """
    plots a spectrom with correct time axis and frequency
    @params:
        spec: (nbins, frames) type Complex spectrogram
        x: audio
        sr: sampling rate
        title: str for title
    """

    t_speech = np.linspace(0, len(x)/sr, spec.shape[1])
    max_freq = np.max(np.fft.fftfreq(len(x), d=1/sr))
    freq_scale_speech = np.linspace(0, max_freq, spec.shape[0])
    fig, ax = plt.subplots(figsize=(16,9))
    ax.title.set_text(title)
    ax.set(xlabel="time (s)", ylabel= "Frequency (Hz)")
    ax.pcolormesh(t_speech,freq_scale_speech,np.power(np.abs(spec),0.4))
    return


def stft_frame_to_time(idx, s, dft_size, sr):
    """
    STFT spec has shape (nbins, frames)
    
    say we are given frame_i and want to map back to second
    
    ie we want to convert frame 3 to time
    
    [frame_0]
        [frame_1]
    | s |   [frame_2]
        | s |    [frame_3]
            | s  |
            
    So there are 3*s 
    
    Formula:  ((i-1)*s+dft_size)/sr = time 
            
    @params:
        idx: idx in frames in (nbins, frames)
        s: hop size of STFT
        dft_size: yes
        sr: yes
    returns:
        time in seconds
    """
    return ((i-1)*s+dft_size)/sr 

def stft_time_to_frame(t, s, dft_size, sr):
    """
    inverse of the one above
    
    in the inverse case if u look at the formula above, if t = 0 for this function it would not work
    hence we have to "gate it"
    
    @params:
        t: time in seconds
        everything else the same 
    """
    if t == 0:
        return 0 
    else:
        return int((t*sr-dft_size)/s +1)


def sound( x, rate=8000, label=''):
    from IPython.display import display, Audio, HTML
    display( HTML( 
    '<style> table, th, td {border: 0px; }</style> <table><tr><td>' + label + 
    '</td><td>' + Audio( x, rate=rate)._repr_html_()[3:] + '</td></tr></table>'
    ))