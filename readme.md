posts에 넣고


---
title: "Diffusion Model : DDPM 정리"
date: 2023-02-17
readingTime: 20 
thumbnail: images/post-001-thumb.jpg
tags: [Generative AI, Diffusion, DDPM]
---


npm run build
npm run preview   # 브라우저로 http://localhost:5000 등 확인

로컬 posts/ 에 글.md 추가
        ↓
git add .  &&  git commit -m "Add: 새 글"
        ↓
git push origin main
        ↓
GitHub Actions
   • npm ci
   • npm run build       # dist/ 생성
   • dist/ → gh-pages 브랜치 커밋
        ↓
https://san9min.github.io/   ← 1-2분 후 새 글/카드 표시
