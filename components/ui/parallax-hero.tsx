"use client";

import React, { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

export interface ParallaxLayer {
  src: string;
  alt: string;
  speedX: number;
  speedY: number;
  speedZ: number;
  rotation: number;
  distance: number;
  className?: string;
  zIndex: number;
  initialTop: string;
  initialLeft: string;
  width: string;
}

export interface ParallaxHeroProps {
  layers?: ParallaxLayer[];
  title?: string;
  className?: string;
  children?: React.ReactNode;
}

const defaultLayers: ParallaxLayer[] = [
  {
    src: "/static/parallax/background.png",
    alt: "background",
    speedX: 0.03,
    speedY: 0.038,
    speedZ: 0,
    rotation: 0,
    distance: -200,
    zIndex: 1,
    initialTop: "calc(50% - 50px)",
    initialLeft: "calc(50% + 0px)",
    width: "3200px",
  },
  {
    src: "/static/parallax/fog-7.png",
    alt: "fog-7",
    speedX: 0.27,
    speedY: 0.32,
    speedZ: 0,
    rotation: 0,
    distance: 850,
    zIndex: 2,
    initialTop: "calc(50% - 100px)",
    initialLeft: "calc(50% + 300px)",
    width: "1900px",
  },
  {
    src: "/static/parallax/mountain-10.png",
    alt: "mountain-10",
    speedX: 0.095,
    speedY: 0.005,
    speedZ: 0,
    rotation: 0,
    distance: 1110,
    zIndex: 3,
    initialTop: "calc(50% + 169px)",
    initialLeft: "calc(50% + 330px)",
    width: "1200px",
  },
  {
    src: "/static/parallax/fog-6.png",
    alt: "fog-6",
    speedX: 0.25,
    speedY: 0.28,
    speedZ: 0,
    rotation: 0,
    distance: 1400,
    zIndex: 4,
    initialTop: "calc(50% + 285px)",
    initialLeft: "calc(50%)",
    width: "2200px",
    className: "opacity-30",
  },
  {
    src: "/static/parallax/mountain-9.png",
    alt: "mountain-9",
    speedX: 0.125,
    speedY: 0.155,
    speedZ: 0.15,
    rotation: 0.02,
    distance: 1700,
    zIndex: 5,
    initialTop: "calc(50% + 313px)",
    initialLeft: "calc(50% - 557px)",
    width: "670px",
  },
  {
    src: "/static/parallax/fog-5.png",
    alt: "fog-5",
    speedX: 0.16,
    speedY: 0.105,
    speedZ: 0,
    rotation: 0,
    distance: 1900,
    zIndex: 6,
    initialTop: "calc(50% + 360px)",
    initialLeft: "calc(50% + 40px)",
    width: "650px",
  },
  {
    src: "/static/parallax/mountain-7.png",
    alt: "mountain-7",
    speedX: 0.1,
    speedY: 0.1,
    speedZ: 0,
    rotation: 0.09,
    distance: 2000,
    zIndex: 7,
    initialTop: "calc(50% + 223px)",
    initialLeft: "calc(50% + 495px)",
    width: "738px",
  },
  {
    src: "/static/parallax/mountain-6.png",
    alt: "mountain-6",
    speedX: 0.065,
    speedY: 0.05,
    speedZ: 0.05,
    rotation: 0.12,
    distance: 2300,
    zIndex: 8,
    initialTop: "calc(50% + 120px)",
    initialLeft: "calc(50% + 590px)",
    width: "408px",
  },
  {
    src: "/static/parallax/fog-4.png",
    alt: "fog-4",
    speedX: 0.135,
    speedY: 0.1,
    speedZ: 0,
    rotation: 0,
    distance: 2400,
    zIndex: 9,
    initialTop: "calc(50% + 223px)",
    initialLeft: "calc(50% + 460px)",
    width: "590px",
    className: "opacity-50",
  },
  {
    src: "/static/parallax/mountain-5.png",
    alt: "mountain-5",
    speedX: 0.08,
    speedY: 0.05,
    speedZ: 0.13,
    rotation: 0.1,
    distance: 2550,
    zIndex: 10,
    initialTop: "calc(50% + 320px)",
    initialLeft: "calc(50% + 230px)",
    width: "725px",
  },
  {
    src: "/static/parallax/fog-3.png",
    alt: "fog-3",
    speedX: 0.11,
    speedY: 0.018,
    speedZ: 0,
    rotation: 0,
    distance: 2800,
    zIndex: 11,
    initialTop: "calc(50% + 210px)",
    initialLeft: "calc(50% + 5px)",
    width: "1600px",
  },
  {
    src: "/static/parallax/mountain-4.png",
    alt: "mountain-4",
    speedX: 0.059,
    speedY: 0.024,
    speedZ: 0.35,
    rotation: 0.14,
    distance: 3200,
    zIndex: 12,
    initialTop: "calc(50% + 196px)",
    initialLeft: "calc(50% - 698px)",
    width: "1100px",
  },
  {
    src: "/static/parallax/mountain-3.png",
    alt: "mountain-3",
    speedX: 0.04,
    speedY: 0.018,
    speedZ: 0.32,
    rotation: 0.05,
    distance: 3400,
    zIndex: 13,
    initialTop: "calc(50% - 20px)",
    initialLeft: "calc(50% + 750px)",
    width: "630px",
  },
  {
    src: "/static/parallax/fog-2.png",
    alt: "fog-2",
    speedX: 0.15,
    speedY: 0.0115,
    speedZ: 0,
    rotation: 0,
    distance: 3600,
    zIndex: 14,
    initialTop: "calc(50% - 20px)",
    initialLeft: "calc(50% + 698px)",
    width: "1100px",
  },
  {
    src: "/static/parallax/mountain-2.png",
    alt: "mountain-2",
    speedX: 0.0235,
    speedY: 0.013,
    speedZ: 0.42,
    rotation: 0.15,
    distance: 3800,
    zIndex: 15,
    initialTop: "calc(50% + 256px)",
    initialLeft: "calc(50% + 528px)",
    width: "800px",
  },
  {
    src: "/static/parallax/mountain-1.png",
    alt: "mountain-1",
    speedX: 0.027,
    speedY: 0.018,
    speedZ: 0.53,
    rotation: 0.2,
    distance: 4000,
    zIndex: 16,
    initialTop: "calc(50% + 196px)",
    initialLeft: "calc(50% - 728px)",
    width: "1100px",
  },
  {
    src: "/static/parallax/fog-1.png",
    alt: "fog-1",
    speedX: 0.12,
    speedY: 0.01,
    speedZ: 0,
    rotation: 0,
    distance: 4200,
    zIndex: 17,
    initialTop: "calc(100% - 355px)",
    initialLeft: "calc(50% + 100px)",
    width: "1900px",
    className: "opacity-50",
  },
];

export const ParallaxHero: React.FC<ParallaxHeroProps> = ({
  layers = defaultLayers,
  title = "MIKU AI",
  className,
  children,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const layerRefs = useRef<(HTMLImageElement | null)[]>([]);
  const textRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const newXValue = e.clientX - window.innerWidth / 2;
      const newYValue = e.clientY - window.innerHeight / 2;
      const newRotateDegree = (newXValue / (window.innerWidth / 2)) * 20;

      layerRefs.current.forEach((el, index) => {
        if (!el) return;
        const layer = layers[index];
        const { speedX, speedY, speedZ, rotation } = layer;
        const computedLeft = parseFloat(getComputedStyle(el).left.replace("px", ""));
        const isInLeft = computedLeft < window.innerWidth / 2 ? 1 : -1;
        const zValue = (e.clientX - computedLeft) * isInLeft * 0.1;

        el.style.transform = `perspective(2300px) translateZ(${
          zValue * speedZ
        }px) rotateY(${newRotateDegree * rotation}deg) translateX(calc(-50% + ${
          -newXValue * speedX
        }px)) translateY(calc(-50% + ${newYValue * speedY}px))`;
      });

      if (textRef.current) {
        const textSpeedX = 0.07;
        const textSpeedY = 0.05;
        const textSpeedZ = 0.08;
        const textRotation = 0.04;
        const computedLeft = parseFloat(getComputedStyle(textRef.current).left.replace("px", ""));
        const isInLeft = computedLeft < window.innerWidth / 2 ? 1 : -1;
        const zValue = (e.clientX - computedLeft) * isInLeft * 0.1;

        textRef.current.style.transform = `perspective(2300px) translateZ(${
          zValue * textSpeedZ
        }px) rotateY(${newRotateDegree * textRotation}deg) translateX(calc(-50% + ${
          -newXValue * textSpeedX
        }px)) translateY(calc(-50% + ${newYValue * textSpeedY}px))`;
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [layers]);

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative min-h-screen w-full overflow-hidden bg-slate-950 font-sans text-slate-100",
        className
      )}
    >
      {/* 3D Parallax Background (Fixed & Non-blocking) */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute inset-0 z-[50] bg-[radial-gradient(ellipse_at_center,rgba(15,23,42,0.45)_0%,rgba(3,7,18,0.9)_100%)]" />

        {layers.map((layer, index) => (
          <img
            key={index}
            ref={(el) => {
              layerRefs.current[index] = el;
            }}
            src={layer.src}
            alt={layer.alt}
            className={cn("absolute pointer-events-none transition-transform duration-[450ms] ease-out", layer.className)}
            style={{
              width: layer.width,
              top: layer.initialTop,
              left: layer.initialLeft,
              zIndex: layer.zIndex,
              transform: "translate(-50%, -50%)",
            }}
          />
        ))}

        {title && (
          <div
            ref={textRef}
            className="absolute z-[25] text-slate-400/10 font-black tracking-widest uppercase select-none text-center transition-transform duration-[450ms] ease-out"
            style={{
              top: "calc(50% - 140px)",
              left: "50%",
              transform: "translate(-50%, -50%)",
            }}
          >
            <span className="text-[12rem] leading-none max-lg:text-[8rem] max-md:text-[4.5rem]">
              {title}
            </span>
          </div>
        )}
      </div>

      {/* Foreground Content (Interactive App Layer) */}
      <div className="relative z-10 w-full min-h-screen flex flex-col p-4 md:p-6 lg:p-8 pointer-events-auto">
        {children}
      </div>
    </div>
  );
};

export default ParallaxHero;
