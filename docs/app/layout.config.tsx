import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";

export const baseOptions: BaseLayoutProps = {
  nav: {
    title: (
      <span style={{ fontFamily: "'SF Mono', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', ui-monospace, monospace", fontWeight: 700, letterSpacing: "0.05em" }}>
        <span style={{ color: "#66cc99" }}>humux</span>
      </span>
    ),
    url: "/docs",
  },
  links: [
    {
      text: "GitHub",
      url: "https://github.com/mattmezza/humux",
    },
  ],
  githubUrl: "https://github.com/mattmezza/humux",
};
