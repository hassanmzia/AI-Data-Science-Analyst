/// <reference types="react-scripts" />

declare module 'react-plotly.js' {
  import { Component } from 'react';
  interface PlotParams {
    data: any[];
    layout?: any;
    config?: any;
    style?: React.CSSProperties;
    className?: string;
  }
  export default class Plot extends Component<PlotParams> {}
}
