import "./globals_dark.css";

export const metadata = {
  title: "Dehook Maestro",
  description: "Class Final Project for Accessible Computing of Spring 2023",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
