from __future__ import annotations
 
import os
import signal
from typing import Any, Callable, Dict, List, Optional
 
from BaseLayer import BaseLayer
 
 
_SUPPORTED_SIGNALS = {signal.SIGINT, signal.SIGTERM}   # available everywhere
 
if os.name != "nt":
    # Linux / macOS only
    _SUPPORTED_SIGNALS.update({
        signal.SIGUSR1,   # SLURM / PBS / SGE walltime warning
        signal.SIGUSR2,   # LSF / SGE secondary warning
        signal.SIGHUP,    # SSH session lost
    })
class SignalLayer(BaseLayer):
    """
    Listens for OS signals sent by the HPC scheduler and notifies
    the AutonomousController immediately so it can decide what to do.
 
    This layer has no state of its own — it does not track any variables,
    does not contribute to snapshots, and has nothing to restore.
 
    Its only job is:
        1. Catch incoming OS signals
        2. Notify the AC immediately with the signal name
        3. The AC decides whether to checkpoint
 
    Typical setup (done by RunnerScript, never by the user):
        signal_layer = SignalLayer(
            poll_target=[],
            snapshot_target=[],
            ac_notify_callback=autonomous_controller.on_signal_received,
        )
    """

    def __init__(
        self,
        poll_target: List[str],
        snapshot_target: List[str],
        ac_notify_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        poll_target:
            Always empty [] — SignalLayer tracks no variables.
        snapshot_target:
            Always empty [] — SignalLayer tracks no variables.
        ac_notify_callback:
            Called immediately when any signal fires.
            Signature: callback(signal_name)
            Should point to AutonomousController.on_signal_received().
            The AC then decides whether and how to checkpoint.
        """
        super().__init__(poll_target,snapshot_target)
        self._ac_notify_callback=ac_notify_callback
        self._original_handlers:Dict[signal.Signals,Any]={} # original OS handlers — restored on detach()

    def attach(self)->None:
        for sig in _SUPPORTED_SIGNALS:
            try:
                self._original_handlers[sig]=signal.getsignal(sig)
                signal.signal(sig,self._on_signal)
            except (OSError,ValueError):
                # some signals cannot be caught in certain environments
                # skip silently rather than crashing at startup
                pass
        self._set_active(True)
    
    def detach(self)->None:
        """
        Restore all original signal handlers.
        Called when training ends or AutoCheck shuts down.
        """
        for sig, original_handlers in self._original_handlers.items():
            try:
                signal.signal(sig,original_handlers)
            except(OSError,ValueError):
                pass
        self._original_handlers.clear()
        self._set_active(False)

    def poll(self)->Dict[str,Any]:
        """
        SignalLayer does not use poll() — all signals are reported
        to the AC immediately via ac_notify_callback.
        Always returns {}.
        """
        return {}
    def snapshot(self) -> Dict[str, Any]:
        """
        SignalLayer has no state to snapshot.
        Always returns {}.
        """
        return {}
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """
        SignalLayer has no state to restore.
        Signal handlers are re-registered fresh by attach() on every run.
        """
        pass

    #Signal handler
    def _on_signal(self,signum:int, frame:Any)->None:
        #Fired by the OS when any registered signal is received.
        signal_name=signal.Signals(signum).name

        if self._ac_notify_callback is not None:
            self._ac_notify_callback(signal_name)