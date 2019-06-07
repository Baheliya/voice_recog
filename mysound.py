# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 23:41:26 2019

@author: lshri
"""

"""Plots
Time in MS Vs Amplitude in DB of a input wav signal
"""

import numpy
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft




myAudio = "Recording6.wav"

#Read file and get sampling freq [ usually 44100 Hz ]  and sound object
samplingFreq, mySound = wavfile.read(myAudio)

#Check if wave file is 16bit or 32 bit. 24bit is not supported
mySoundDataType = mySound.dtype



#We can convert our sound array to floating point values ranging from -1 to 1 as follows

mySound = mySound / (2.**15)
ms  =mySound
print(mySound.shape)
#Check sample points and sound channel for duel channel(5060, 2) or  (5060, ) for mono channel
#if(mySound.shape==samplePoints,2):
"""use   if   stereo   
 mySound = mySound[:,1]"""


mySound = mySound[:,1]

mySoundShape = mySound.shape
samplePoints = float(mySound.shape[0])

#Get duration of sound file
signalDuration =  mySound.shape[0] / samplingFreq
#for   500  milliseconds

milsec =int(input("Enter minimum silence threshold  in  milliseconds  "))
gap = (samplePoints)*milsec/(signalDuration *1000)
gap =int(gap)
#print(gap)
#print(mySound.shape)
top = numpy.mean(numpy.abs(mySound))
for i in range(mySound.shape[0]):
    if numpy.abs(mySound[i]) < top*4:
        mySound[i]=0
#If two channels, then select only one channel
mySoundOneChannel = mySound
mss =mySound   * (2.**15)

#wavfile.write('ch.wav', samplingFreq, mss)
#Plotting the tone

# We can represent sound by plotting the pressure values against time axis.
#Create an array of sample point in one dimension
timeArray = numpy.arange(0, samplePoints, 1)

#
timeArray = timeArray / samplingFreq

#Scale to milliSeconds
timeArray = timeArray *1000

#Plot the tone
plt.plot(timeArray, ms, color='G')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Original')
plt.show()


plt.plot(timeArray, mySoundOneChannel, color='G')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Supressed  noise')
plt.show()

def  is_gap(mySound,index,gap):
    flag =True
    for i in range(gap):
        if(mySound[i+index]!=0):
            flag = False
        
    return flag,i
number =0
def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = numpy.concatenate(([0], numpy.equal(a, 0).view(numpy.int8), [0]))
    absdiff = numpy.abs(numpy.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = numpy.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

ranges = zero_runs(mySound)
#print(ranges)
for i  in  range(ranges.shape[0]):
    #print(numpy.abs(ranges[i][1]  -  ranges[i][0]))
    if  gap <= numpy.abs(ranges[i][1]  -  ranges[i][0]):
        print((ranges[i]*1000/samplingFreq))
        number =  number  +  1
print("The of  silence  gaps  are",number)

#Plot frequency content
#We can get frquency from amplitude and time using FFT , Fast Fourier Transform algorithm

#Get length of mySound object array
mySoundLength = len(mySound)

#Take the Fourier transformation on given sample point 
#fftArray = fft(mySound)
fftArray = fft(mySoundOneChannel)

numUniquePoints = numpy.ceil((mySoundLength + 1) / 2.0)
fftArray = fftArray[0:int(numUniquePoints)]

#FFT contains both magnitude and phase and given in complex numbers in real + imaginary parts (a + ib) format.
#By taking absolute value , we get only real part

fftArray = abs(fftArray)

#Scale the fft array by length of sample points so that magnitude does not depend on
#the length of the signal or on its sampling frequency

fftArray = fftArray / float(mySoundLength)

#FFT has both positive and negative information. Square to get positive only
fftArray = fftArray **2

#Multiply by two (research why?)
#Odd NFFT excludes Nyquist point
if mySoundLength % 2 > 0: #we've got odd number of points in fft
    fftArray[1:len(fftArray)] = fftArray[1:len(fftArray)] * 2

else: #We've got even number of points in fft
    fftArray[1:len(fftArray) -1] = fftArray[1:len(fftArray) -1] * 2  

freqArray = numpy.arange(0, numUniquePoints, 1.0) * (samplingFreq / mySoundLength);

#Plot the frequency
plt.plot(freqArray/1000, 10 * numpy.log10 (fftArray), color='B')
plt.xlabel('Frequency (Khz)')
plt.ylabel('Power (dB)')
plt.show()

#Get List of element in frequency array
#print freqArray.dtype.type
freqArrayLength = len(freqArray)
print ("freqArrayLength =", freqArrayLength)
numpy.savetxt("freqData.txt", freqArray, fmt='%6.2f')

#Print FFtarray information
print ("fftArray length =", len(fftArray))
numpy.savetxt("fftData.txt", fftArray)


from pydub import AudioSegment
from pydub.silence import detect_silence,split_on_silence
sound_file = AudioSegment.from_wav("Recording6.wav")
audio_chunks = detect_silence(sound_file, 
    # must be silent for at least half a second
    min_silence_len=200,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-70
)
print("The   number   of   words    are",len(audio_chunks)-1)
print("Silence  intervals",audio_chunks)