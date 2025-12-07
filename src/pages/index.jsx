import React from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import HeroImage from '@site/src/assets/cover.png';

export default function Home() {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout
      title={siteConfig.title}
      description="Physical AI and Humanoid Robotics – a complete roadmap to embodied intelligence, simulation, and humanoid systems">
      <main className="landing-root">
        {/* Hero Section */}
        <section className="landing-hero">
          <div className="landing-hero-inner">
            <div className="landing-hero-copy">
              <p className="landing-eyebrow">Digital Textbook Portal</p>
              <h1 className="landing-title">Physical AI and Humanoid Robotics</h1>
              <p className="landing-subtitle">
                A complete roadmap to embodied intelligence, simulation, and humanoid systems.
              </p>

              <p className="landing-description">
                Master the full stack of Physical AI: from ROS&nbsp;2 robot control and Gazebo/Unity simulation to
                NVIDIA Isaac workflows, humanoid kinematics, and cutting-edge Vision-Language-Action (VLA) systems.
                This textbook guides you from first principles to deployment-ready humanoid robotics.
              </p>

              <div className="landing-actions-row">
                <div className="landing-cta-group">
                  <Link
                    className={clsx('button button--lg landing-btn-primary')}
                    to="/ch1-introduction-to-physical-ai">
                    Start Reading
                  </Link>
                 
                </div>

                <div className="landing-stat-badge" aria-label="Total Chapters">
                  <span className="landing-stat-label">Total Chapters</span>
                  <span className="landing-stat-value">13</span>
                  <span className="landing-stat-unit">Chapters</span>
                </div>
              </div>
            </div>

            <div>
             
              <img src={HeroImage} alt="Physical AI and humanoid robotics illustration" className="landing-hero-image" />
              
            </div>
          </div>
        </section>

        {/* Coverage Section */}
        <section className="landing-section landing-coverage">
          <div className="landing-section-header">
            <h2 className="landing-section-title">What this book covers</h2>
            <p className="landing-section-subtitle">
              A tightly structured roadmap across the core layers of modern humanoid robotics and Physical AI.
            </p>
          </div>

          <div className="landing-coverage-grid">
            <article className="landing-card">
              <div className="landing-card-icon" aria-hidden="true"></div>
              <h3 className="landing-card-title">ROS&nbsp;2 &amp; robot control</h3>
              <p className="landing-card-body">
                Build robust control stacks, message graphs, and real-time behaviors on top of ROS&nbsp;2.
              </p>
            </article>

            <article className="landing-card">
              <div className="landing-card-icon" aria-hidden="true"></div>
              <h3 className="landing-card-title">Gazebo &amp; Unity simulation</h3>
              <p className="landing-card-body">
                Prototype, test, and validate robots in photorealistic and physics-accurate virtual worlds.
              </p>
            </article>

            <article className="landing-card">
              <div className="landing-card-icon" aria-hidden="true"></div>
              <h3 className="landing-card-title">NVIDIA Isaac platform</h3>
              <p className="landing-card-body">
                Leverage GPU-accelerated simulation, perception, and policy learning with Isaac tools.
              </p>
            </article>

            <article className="landing-card">
              <div className="landing-card-icon" aria-hidden="true"></div>
              <h3 className="landing-card-title">Humanoid engineering</h3>
              <p className="landing-card-body">
                Design mechanisms, kinematics, and whole-body control for bipedal humanoid systems.
              </p>
            </article>

            <article className="landing-card">
              <div className="landing-card-icon" aria-hidden="true"></div>
              <h3 className="landing-card-title">Vision-Language-Action</h3>
              <p className="landing-card-body">
                Connect perception, language, and action policies for grounded, goal-directed behavior.
              </p>
            </article>
          </div>
        </section>
      </main>
    </Layout>
  );
}
