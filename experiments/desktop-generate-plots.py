import sys
sys.path.append('../src')
import policies
import bbDebiasing
import maxEnsembleDebias

sys.path.append('experiment-pipeline')

import analysisHelp

experimentName='poly-label_gb_coord_variance'

bbPath = 'experiment-pipeline/debiased-models/desktop-experiments/{0}_BBModel.pkl'.format(experimentName)
maxEnsemblePath = 'experiment-pipeline/debiased-models/desktop-experiments/{0}_MaxEnsemble.pkl'.format(experimentName)
figPath='experiment-pipeline/figs/desktop-experiments/{0}'.format(experimentName)

analysisHelp.allPlots(bbPath, maxEnsemblePath, figPath=figPath, experimentName=experimentName)
