# sources: 
#   - http://code.activestate.com/recipes/65287-automatically-start-the-debugger-on-an-exception/
#   - http://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
import sys, pdb # you want pdb to be available after import

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print()
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)
        # from IPython import embed; embed()

sys.excepthook = info
