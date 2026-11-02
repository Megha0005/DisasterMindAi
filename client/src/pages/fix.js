const fs = require("fs");
const file = process.argv[2];
let s = fs.readFileSync(file, "utf8");

const bad1 = "\u00EF\u00BF\u00BD"; // the text "ï¿½"
const bad2 = "\uFFFD"; // the real replacement char "�"

for (const bad of [bad1, bad2]) {
  s = s
    .split("2" + bad + " " + bad + " 2" + bad)
    .join("2\u00B0 \u00D7 2\u00B0");
  s = s.split(bad).join("\u00B0");
}

fs.writeFileSync(file, s, "utf8");
console.log("Done:", file);
