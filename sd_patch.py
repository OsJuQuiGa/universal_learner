import sounddevice as sd
sd._last_callback_play = None 
sd._last_callback_rec = None
def stop(ignore_errors=True, callback=None):
    if callback:
        callback.stream.stop(ignore_errors)
        callback.stream.close(ignore_errors)
    else:
        sd.stop()

class _CallbackContext(sd._CallbackContext):
    def start_stream(self, StreamClass, samplerate, channels, dtype, callback,
                        blocking, **kwargs):
        if StreamClass == sd.OutputStream:
            last_callback = sd._last_callback_play
        elif StreamClass == sd.InputStream:
            last_callback = sd._last_callback_rec
        else:
            last_callback = None

        if last_callback:
            stop(callback = last_callback)
        else:
            sd.stop()
        # Stop previous playback/recording
        self.stream = StreamClass(samplerate=samplerate,
                                    channels=channels,
                                    dtype=dtype,
                                    callback=callback,
                                    finished_callback=self.finished_callback,
                                    **kwargs)
        self.stream.start()
        #global _last_callback
        if StreamClass == sd.OutputStream:
            sd._last_callback_play = self
        elif StreamClass == sd.InputStream:
            sd._last_callback_rec = self
        else:
            sd._last_callback = self

        if blocking:
            self.wait()
#class _CallbackContext(sd._CallbackContext):
#    pass
sd._CallbackContext = _CallbackContext
