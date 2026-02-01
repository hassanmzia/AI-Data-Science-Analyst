/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_MCP_URL: string;
  readonly VITE_WS_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module 'plotly.js-dist-min' {
  import Plotly from 'plotly.js';
  export default Plotly;
}

declare module 'react-plotly.js/factory' {
  import { ComponentType } from 'react';
  function createPlotlyComponent(plotly: any): ComponentType<any>;
  export default createPlotlyComponent;
}
