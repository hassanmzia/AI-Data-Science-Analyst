import createPlotlyComponent from 'react-plotly.js/factory';
// @ts-ignore - plotly.js-dist-min doesn't have types but works at runtime
import Plotly from 'plotly.js-dist-min';

const Plot = createPlotlyComponent(Plotly);

export default Plot;
