import nextConfig from "eslint-config-next";

const config = [
  {
    ignores: [".venv-yolo/**", ".model-export/**", "public/models/**/*.onnx"],
  },
  ...nextConfig,
];

export default config;
