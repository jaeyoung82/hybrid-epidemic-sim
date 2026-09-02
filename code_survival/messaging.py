"""
Output control for survival analysis.

Provides utilities to control verbosity and suppress mediation metrics in default output.
"""

def should_print_mediation() -> bool:
    """
    Check whether mediation metrics should be printed.
    
    Returns False by default to suppress mediation output in CLI.
    Set SURVIVAL_ANALYSIS_VERBOSE=1 in environment to enable.
    """
    import os
    return os.environ.get('SURVIVAL_ANALYSIS_VERBOSE', '0') == '1'