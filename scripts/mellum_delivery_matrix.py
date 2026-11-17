"""Single source of truth for final cumulative rows and the delivery launcher."""
ROWS={
 'step0':{'profile':'delivery_step0','slots':1,'parent':None,'points':[]},
 'step01':{'profile':'delivery_step01','slots':32,'parent':'step0','points':[1]},
 'step012':{'profile':'delivery_step012','slots':32,'parent':'step01','points':[1,2]},
 'step0123':{'profile':'delivery_step0123','slots':32,'parent':'step012','points':[1,2,3]},
}
DEFAULT_ROW='step0123'

def slots(profile):
    return next((v['slots'] for v in ROWS.values() if v['profile']==profile),32)
