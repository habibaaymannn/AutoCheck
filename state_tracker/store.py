class StateStore:

    def __init__(self, tracked_vars=None, allow_dynamic=True):
        
        if tracked_vars is None:
            tracked_vars = [] 
        self.tracked_vars = tracked_vars
        self.allow_dynamic = allow_dynamic
        self.current_state = {var: None for var in tracked_vars}

   
    def set(self, key, value):
        
        if key in self.tracked_vars or self.allow_dynamic:
            self.current_state[key] = value
            if self.allow_dynamic and key not in self.tracked_vars:
                self.tracked_vars.append(key)
        else:
            raise KeyError(f"Variable '{key}' is not tracked.")

    
    def get(self, key, default=None):
       
        if key in self.current_state:
            return self.current_state[key]
        elif self.allow_dynamic:
            self.current_state[key] = default
            self.tracked_vars.append(key)
            return default
        else:
            raise KeyError(f"Variable '{key}' is not tracked.")

  
    def remove(self, key):
       
        if key in self.current_state:
            self.current_state.pop(key)
            if key in self.tracked_vars:
                self.tracked_vars.remove(key)
        else:
            raise KeyError(f"Variable '{key}' not found in state.")

    
    def reset(self):
        
        for var in self.tracked_vars:
            self.current_state[var] = None

   
    def all(self):
        
        return self.current_state.copy()