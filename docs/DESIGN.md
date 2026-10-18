# PlateBack interface

The October 17 refresh takes cues from [Ramp](https://ramp.com/)'s restrained navigation and bright action color, [Cluely](https://cluely.com/)'s editorial typography, and [Clay](https://www.clay.com/)'s tactile illustration and generous spacing. It uses an original SVG plate illustration; no reference-site logos, images, or page layouts were copied.

The design uses warm paper (#f8f8f2), near-black ink (#24261f), muted olive-gray supporting text, fine borders, and lime (#e0f064) primary actions. Instrument Serif is reserved for page headings and large readings; DM Sans handles controls and explanatory text. Local Georgia/Arial fallbacks keep the interface usable when Google Fonts cannot load. Food mask colors remain distinct from interface action colors.

Navigation is horizontal and moves to a second row on small screens. The capture/review panel and collection controls stack below 720px. Buttons retain text labels, controls have visible keyboard focus, reduced-motion settings remove transitions, and the page has no horizontal overflow at a 390px viewport. Dashboard metrics distinguish unknown weight from a measured zero. Demo data stays excluded by default. Model limitations remain visible beside review controls.

Browser evidence: `output/playwright/redesign-desktop.png`, `redesign-mobile.png`, and `redesign-dashboard.png`. These show the running application, not mockups. Functionality is verified separately in the application verification record.
