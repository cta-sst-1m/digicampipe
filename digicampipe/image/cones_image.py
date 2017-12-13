





<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
  <link rel="dns-prefetch" href="https://assets-cdn.github.com">
  <link rel="dns-prefetch" href="https://avatars0.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars1.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars2.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars3.githubusercontent.com">
  <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">
  <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">



  <link crossorigin="anonymous" href="https://assets-cdn.github.com/assets/frameworks-1c2c316b7a17f15536c6a26ed744654fc228a24658a7ae395cdcbf8329c8406b.css" media="all" rel="stylesheet" />
  <link crossorigin="anonymous" href="https://assets-cdn.github.com/assets/github-9220ee04047cbb7ee0ee683d508378700cfaf21fbd3eea2b46ae6f847bd16e07.css" media="all" rel="stylesheet" />
  
  
  <link crossorigin="anonymous" href="https://assets-cdn.github.com/assets/site-44b6bba3881278f33c221b6526379b55fbd098af3e553f54e81cab4c9a517c8e.css" media="all" rel="stylesheet" />
  

  <meta name="viewport" content="width=device-width">
  
  <title>digicampipe/cones_image.py at lid_cdd_analysis · calispac/digicampipe · GitHub</title>
  <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub">
  <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
  <meta property="fb:app_id" content="1401488693436528">

    
    <meta content="https://avatars2.githubusercontent.com/u/23462345?s=400&amp;v=4" property="og:image" /><meta content="GitHub" property="og:site_name" /><meta content="object" property="og:type" /><meta content="calispac/digicampipe" property="og:title" /><meta content="https://github.com/calispac/digicampipe" property="og:url" /><meta content="digicampipe - DigiCam pipeline based on ctapipe" property="og:description" />

  <link rel="assets" href="https://assets-cdn.github.com/">
  
  <meta name="pjax-timeout" content="1000">
  
  <meta name="request-id" content="B93A:074B:11C0CFA:2096ED7:5A3135A8" data-pjax-transient>
  

  <meta name="selected-link" value="repo_source" data-pjax-transient>

    <meta name="google-site-verification" content="KT5gs8h0wvaagLKAVWq8bbeNwnZZK1r1XQysX3xurLU">
  <meta name="google-site-verification" content="ZzhVyEFwb7w3e0-uOTltm8Jsck2F5StVihD0exw2fsA">
  <meta name="google-site-verification" content="GXs5KoUUkNCoaAZn7wPN-t01Pywp9M3sEjnt_3_ZWPc">
    <meta name="google-analytics" content="UA-3769691-2">

<meta content="collector.githubapp.com" name="octolytics-host" /><meta content="github" name="octolytics-app-id" /><meta content="https://collector.githubapp.com/github-external/browser_event" name="octolytics-event-url" /><meta content="B93A:074B:11C0CFA:2096ED7:5A3135A8" name="octolytics-dimension-request_id" /><meta content="iad" name="octolytics-dimension-region_edge" /><meta content="iad" name="octolytics-dimension-region_render" />
<meta content="/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show" data-pjax-transient="true" name="analytics-location" />




  <meta class="js-ga-set" name="dimension1" content="Logged Out">


  

      <meta name="hostname" content="github.com">
  <meta name="user-login" content="">

      <meta name="expected-hostname" content="github.com">
    <meta name="js-proxy-site-detection-payload" content="NDhkMjkwNTk1ZWY2OGI1YTdjYmU2MWZhNjA4ZWEzNjVmNmEzOWUxMDQ3YWIyNWVkNWQ0NjAzMjZiMDI0MGFmNXx7InJlbW90ZV9hZGRyZXNzIjoiMTI5LjE5NC41NS4xMzMiLCJyZXF1ZXN0X2lkIjoiQjkzQTowNzRCOjExQzBDRkE6MjA5NkVENzo1QTMxMzVBOCIsInRpbWVzdGFtcCI6MTUxMzE3NDQ0MSwiaG9zdCI6ImdpdGh1Yi5jb20ifQ==">


  <meta name="html-safe-nonce" content="a81237aa6389b73deb372a9f3e2638ae7cc4391d">

  <meta http-equiv="x-pjax-version" content="2015d3e0d83d16fca0ab6e2e4c9f72d8">
  

      <link href="https://github.com/calispac/digicampipe/commits/lid_cdd_analysis.atom" rel="alternate" title="Recent Commits to digicampipe:lid_cdd_analysis" type="application/atom+xml">

  <meta name="description" content="digicampipe - DigiCam pipeline based on ctapipe">
  <meta name="go-import" content="github.com/calispac/digicampipe git https://github.com/calispac/digicampipe.git">

  <meta content="23462345" name="octolytics-dimension-user_id" /><meta content="calispac" name="octolytics-dimension-user_login" /><meta content="102357407" name="octolytics-dimension-repository_id" /><meta content="calispac/digicampipe" name="octolytics-dimension-repository_nwo" /><meta content="true" name="octolytics-dimension-repository_public" /><meta content="false" name="octolytics-dimension-repository_is_fork" /><meta content="102357407" name="octolytics-dimension-repository_network_root_id" /><meta content="calispac/digicampipe" name="octolytics-dimension-repository_network_root_nwo" /><meta content="false" name="octolytics-dimension-repository_explore_github_marketplace_ci_cta_shown" />


    <link rel="canonical" href="https://github.com/calispac/digicampipe/blob/lid_cdd_analysis/digicampipe/image/cones_image.py" data-pjax-transient>


  <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">

  <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">

  <link rel="mask-icon" href="https://assets-cdn.github.com/pinned-octocat.svg" color="#000000">
  <link rel="icon" type="image/x-icon" class="js-site-favicon" href="https://assets-cdn.github.com/favicon.ico">

<meta name="theme-color" content="#1e2327">



  </head>

  <body class="logged-out env-production page-blob">
    

  <div class="position-relative js-header-wrapper ">
    <a href="#start-of-content" tabindex="1" class="px-2 py-4 show-on-focus js-skip-to-content">Skip to content</a>
    <div id="js-pjax-loader-bar" class="pjax-loader-bar"><div class="progress"></div></div>

    
    
    
      



        <header class="Header header-logged-out  position-relative f4 py-3" role="banner">
  <div class="container-lg d-flex px-3">
    <div class="d-flex flex-justify-between flex-items-center">
      <a class="header-logo-invertocat my-0" href="https://github.com/" aria-label="Homepage" data-ga-click="(Logged out) Header, go to homepage, icon:logo-wordmark">
        <svg aria-hidden="true" class="octicon octicon-mark-github" height="32" version="1.1" viewBox="0 0 16 16" width="32"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
      </a>

    </div>

    <div class="HeaderMenu HeaderMenu--bright d-flex flex-justify-between flex-auto">
        <nav class="mt-0">
          <ul class="d-flex list-style-none">
              <li class="ml-2">
                <a href="/features" class="js-selected-navigation-item HeaderNavlink px-0 py-2 m-0" data-ga-click="Header, click, Nav menu - item:features" data-selected-links="/features /features/project-management /features/code-review /features/project-management /features/integrations /features">
                  Features
</a>              </li>
              <li class="ml-4">
                <a href="/business" class="js-selected-navigation-item HeaderNavlink px-0 py-2 m-0" data-ga-click="Header, click, Nav menu - item:business" data-selected-links="/business /business/security /business/customers /business">
                  Business
</a>              </li>

              <li class="ml-4">
                <a href="/explore" class="js-selected-navigation-item HeaderNavlink px-0 py-2 m-0" data-ga-click="Header, click, Nav menu - item:explore" data-selected-links="/explore /trending /trending/developers /integrations /integrations/feature/code /integrations/feature/collaborate /integrations/feature/ship showcases showcases_search showcases_landing /explore">
                  Explore
</a>              </li>

              <li class="ml-4">
                    <a href="/marketplace" class="js-selected-navigation-item HeaderNavlink px-0 py-2 m-0" data-ga-click="Header, click, Nav menu - item:marketplace" data-selected-links=" /marketplace">
                      Marketplace
</a>              </li>
              <li class="ml-4">
                <a href="/pricing" class="js-selected-navigation-item HeaderNavlink px-0 py-2 m-0" data-ga-click="Header, click, Nav menu - item:pricing" data-selected-links="/pricing /pricing/developer /pricing/team /pricing/business-hosted /pricing/business-enterprise /pricing">
                  Pricing
</a>              </li>
          </ul>
        </nav>

      <div class="d-flex">
          <div class="d-lg-flex flex-items-center mr-3">
            <div class="header-search scoped-search site-scoped-search js-site-search" role="search">
  <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="/calispac/digicampipe/search" class="js-site-search-form" data-scoped-search-url="/calispac/digicampipe/search" data-unscoped-search-url="/search" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /></div>
    <label class="form-control header-search-wrapper js-chromeless-input-container">
        <a href="/calispac/digicampipe/blob/lid_cdd_analysis/digicampipe/image/cones_image.py" class="header-search-scope no-underline">This repository</a>
      <input type="text"
        class="form-control header-search-input js-site-search-focus js-site-search-field is-clearable"
        data-hotkey="s"
        name="q"
        value=""
        placeholder="Search"
        aria-label="Search this repository"
        data-unscoped-placeholder="Search GitHub"
        data-scoped-placeholder="Search"
        autocapitalize="off">
        <input type="hidden" class="js-site-search-type-field" name="type" >
    </label>
</form></div>

          </div>

        <span class="d-inline-block">
            <div class="HeaderNavlink px-0 py-2 m-0">
              <a class="text-bold text-white no-underline" href="/login?return_to=%2Fcalispac%2Fdigicampipe%2Fblob%2Flid_cdd_analysis%2Fdigicampipe%2Fimage%2Fcones_image.py" data-ga-click="(Logged out) Header, clicked Sign in, text:sign-in">Sign in</a>
                <span class="text-gray">or</span>
                <a class="text-bold text-white no-underline" href="/join?source=header-repo" data-ga-click="(Logged out) Header, clicked Sign up, text:sign-up">Sign up</a>
            </div>
        </span>
      </div>
    </div>
  </div>
</header>


  </div>

  <div id="start-of-content" class="show-on-focus"></div>

    <div id="js-flash-container">
</div>



  <div role="main">
        <div itemscope itemtype="http://schema.org/SoftwareSourceCode">
    <div id="js-repo-pjax-container" data-pjax-container>
      





  <div class="pagehead repohead instapaper_ignore readability-menu experiment-repo-nav ">
    <div class="repohead-details-container clearfix container ">

      <ul class="pagehead-actions">
  <li>
      <a href="/login?return_to=%2Fcalispac%2Fdigicampipe"
    class="btn btn-sm btn-with-count tooltipped tooltipped-n"
    aria-label="You must be signed in to watch a repository" rel="nofollow">
    <svg aria-hidden="true" class="octicon octicon-eye" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
    Watch
  </a>
  <a class="social-count" href="/calispac/digicampipe/watchers"
     aria-label="4 users are watching this repository">
    4
  </a>

  </li>

  <li>
      <a href="/login?return_to=%2Fcalispac%2Fdigicampipe"
    class="btn btn-sm btn-with-count tooltipped tooltipped-n"
    aria-label="You must be signed in to star a repository" rel="nofollow">
    <svg aria-hidden="true" class="octicon octicon-star" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74z"/></svg>
    Star
  </a>

    <a class="social-count js-social-count" href="/calispac/digicampipe/stargazers"
      aria-label="0 users starred this repository">
      0
    </a>

  </li>

  <li>
      <a href="/login?return_to=%2Fcalispac%2Fdigicampipe"
        class="btn btn-sm btn-with-count tooltipped tooltipped-n"
        aria-label="You must be signed in to fork a repository" rel="nofollow">
        <svg aria-hidden="true" class="octicon octicon-repo-forked" height="16" version="1.1" viewBox="0 0 10 16" width="10"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
        Fork
      </a>

    <a href="/calispac/digicampipe/network" class="social-count"
       aria-label="1 user forked this repository">
      1
    </a>
  </li>
</ul>

      <h1 class="public ">
  <svg aria-hidden="true" class="octicon octicon-repo" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
  <span class="author" itemprop="author"><a href="/calispac" class="url fn" rel="author">calispac</a></span><!--
--><span class="path-divider">/</span><!--
--><strong itemprop="name"><a href="/calispac/digicampipe" data-pjax="#js-repo-pjax-container">digicampipe</a></strong>

</h1>

    </div>
    
<nav class="reponav js-repo-nav js-sidenav-container-pjax container"
     itemscope
     itemtype="http://schema.org/BreadcrumbList"
     role="navigation"
     data-pjax="#js-repo-pjax-container">

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a href="/calispac/digicampipe/tree/lid_cdd_analysis" class="js-selected-navigation-item selected reponav-item" data-hotkey="g c" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages /calispac/digicampipe/tree/lid_cdd_analysis" itemprop="url">
      <svg aria-hidden="true" class="octicon octicon-code" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M9.5 3L8 4.5 11.5 8 8 11.5 9.5 13 14 8 9.5 3zm-5 0L0 8l4.5 5L6 11.5 2.5 8 6 4.5 4.5 3z"/></svg>
      <span itemprop="name">Code</span>
      <meta itemprop="position" content="1">
</a>  </span>

    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a href="/calispac/digicampipe/issues" class="js-selected-navigation-item reponav-item" data-hotkey="g i" data-selected-links="repo_issues repo_labels repo_milestones /calispac/digicampipe/issues" itemprop="url">
        <svg aria-hidden="true" class="octicon octicon-issue-opened" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"/></svg>
        <span itemprop="name">Issues</span>
        <span class="Counter">7</span>
        <meta itemprop="position" content="2">
</a>    </span>

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a href="/calispac/digicampipe/pulls" class="js-selected-navigation-item reponav-item" data-hotkey="g p" data-selected-links="repo_pulls /calispac/digicampipe/pulls" itemprop="url">
      <svg aria-hidden="true" class="octicon octicon-git-pull-request" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M11 11.28V5c-.03-.78-.34-1.47-.94-2.06C9.46 2.35 8.78 2.03 8 2H7V0L4 3l3 3V4h1c.27.02.48.11.69.31.21.2.3.42.31.69v6.28A1.993 1.993 0 0 0 10 15a1.993 1.993 0 0 0 1-3.72zm-1 2.92c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zM4 3c0-1.11-.89-2-2-2a1.993 1.993 0 0 0-1 3.72v6.56A1.993 1.993 0 0 0 2 15a1.993 1.993 0 0 0 1-3.72V4.72c.59-.34 1-.98 1-1.72zm-.8 10c0 .66-.55 1.2-1.2 1.2-.65 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
      <span itemprop="name">Pull requests</span>
      <span class="Counter">0</span>
      <meta itemprop="position" content="3">
</a>  </span>

    <a href="/calispac/digicampipe/projects" class="js-selected-navigation-item reponav-item" data-hotkey="g b" data-selected-links="repo_projects new_repo_project repo_project /calispac/digicampipe/projects">
      <svg aria-hidden="true" class="octicon octicon-project" height="16" version="1.1" viewBox="0 0 15 16" width="15"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h13a1 1 0 0 0 1-1V1a1 1 0 0 0-1-1z"/></svg>
      Projects
      <span class="Counter" >2</span>
</a>
    <a href="/calispac/digicampipe/wiki" class="js-selected-navigation-item reponav-item" data-hotkey="g w" data-selected-links="repo_wiki /calispac/digicampipe/wiki">
      <svg aria-hidden="true" class="octicon octicon-book" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M3 5h4v1H3V5zm0 3h4V7H3v1zm0 2h4V9H3v1zm11-5h-4v1h4V5zm0 2h-4v1h4V7zm0 2h-4v1h4V9zm2-6v9c0 .55-.45 1-1 1H9.5l-1 1-1-1H2c-.55 0-1-.45-1-1V3c0-.55.45-1 1-1h5.5l1 1 1-1H15c.55 0 1 .45 1 1zm-8 .5L7.5 3H2v9h6V3.5zm7-.5H9.5l-.5.5V12h6V3z"/></svg>
      Wiki
</a>

  <a href="/calispac/digicampipe/pulse" class="js-selected-navigation-item reponav-item" data-selected-links="repo_graphs repo_contributors dependency_graph pulse /calispac/digicampipe/pulse">
    <svg aria-hidden="true" class="octicon octicon-graph" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M16 14v1H0V0h1v14h15zM5 13H3V8h2v5zm4 0H7V3h2v10zm4 0h-2V6h2v7z"/></svg>
    Insights
</a>

</nav>


  </div>

<div class="container new-discussion-timeline experiment-repo-nav">
  <div class="repository-content">

    
  <a href="/calispac/digicampipe/blob/57d647885a442b0c1f88566b8c2a8869e1eb5e2d/digicampipe/image/cones_image.py" class="d-none js-permalink-shortcut" data-hotkey="y">Permalink</a>

  <!-- blob contrib key: blob_contributors:v21:a0108b1909c399fa8c56102a4fcc8f83 -->

  <div class="file-navigation js-zeroclipboard-container">
    
<div class="select-menu branch-select-menu js-menu-container js-select-menu float-left">
  <button class=" btn btn-sm select-menu-button js-menu-target css-truncate" data-hotkey="w"
    title="lid_cdd_analysis"
    type="button" aria-label="Switch branches or tags" aria-expanded="false" aria-haspopup="true">
      <i>Branch:</i>
      <span class="js-select-button css-truncate-target">lid_cdd_analys…</span>
  </button>

  <div class="select-menu-modal-holder js-menu-content js-navigation-container" data-pjax>

    <div class="select-menu-modal">
      <div class="select-menu-header">
        <svg aria-label="Close" class="octicon octicon-x js-menu-close" height="16" role="img" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
        <span class="select-menu-title">Switch branches/tags</span>
      </div>

      <div class="select-menu-filters">
        <div class="select-menu-text-filter">
          <input type="text" aria-label="Filter branches/tags" id="context-commitish-filter-field" class="form-control js-filterable-field js-navigation-enable" placeholder="Filter branches/tags">
        </div>
        <div class="select-menu-tabs">
          <ul>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="branches" data-filter-placeholder="Filter branches/tags" class="js-select-menu-tab" role="tab">Branches</a>
            </li>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="tags" data-filter-placeholder="Find a tag…" class="js-select-menu-tab" role="tab">Tags</a>
            </li>
          </ul>
        </div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="branches" role="menu">

        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/calispac/digicampipe/blob/andrii_v2/digicampipe/image/cones_image.py"
               data-name="andrii_v2"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                andrii_v2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/calispac/digicampipe/blob/avoid_private_ctapipe_fork/digicampipe/image/cones_image.py"
               data-name="avoid_private_ctapipe_fork"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                avoid_private_ctapipe_fork
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/calispac/digicampipe/blob/baseline_script/digicampipe/image/cones_image.py"
               data-name="baseline_script"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                baseline_script
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/calispac/digicampipe/blob/jurysek/digicampipe/image/cones_image.py"
               data-name="jurysek"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jurysek
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open selected"
               href="/calispac/digicampipe/blob/lid_cdd_analysis/digicampipe/image/cones_image.py"
               data-name="lid_cdd_analysis"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                lid_cdd_analysis
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/calispac/digicampipe/blob/look_at_io/digicampipe/image/cones_image.py"
               data-name="look_at_io"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                look_at_io
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/calispac/digicampipe/blob/master/digicampipe/image/cones_image.py"
               data-name="master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/calispac/digicampipe/blob/my_new_feature/digicampipe/image/cones_image.py"
               data-name="my_new_feature"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                my_new_feature
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/calispac/digicampipe/blob/no_LD_LIB_PATH/digicampipe/image/cones_image.py"
               data-name="no_LD_LIB_PATH"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                no_LD_LIB_PATH
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/calispac/digicampipe/blob/slow_data/digicampipe/image/cones_image.py"
               data-name="slow_data"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                slow_data
              </span>
            </a>
        </div>

          <div class="select-menu-no-results">Nothing to show</div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="tags">
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


        </div>

        <div class="select-menu-no-results">Nothing to show</div>
      </div>

    </div>
  </div>
</div>

    <div class="BtnGroup float-right">
      <a href="/calispac/digicampipe/find/lid_cdd_analysis"
            class="js-pjax-capture-input btn btn-sm BtnGroup-item"
            data-pjax
            data-hotkey="t">
        Find file
      </a>
      <button aria-label="Copy file path to clipboard" class="js-zeroclipboard btn btn-sm BtnGroup-item tooltipped tooltipped-s" data-copied-hint="Copied!" type="button">Copy path</button>
    </div>
    <div class="breadcrumb js-zeroclipboard-target">
      <span class="repo-root js-repo-root"><span class="js-path-segment"><a href="/calispac/digicampipe/tree/lid_cdd_analysis"><span>digicampipe</span></a></span></span><span class="separator">/</span><span class="js-path-segment"><a href="/calispac/digicampipe/tree/lid_cdd_analysis/digicampipe"><span>digicampipe</span></a></span><span class="separator">/</span><span class="js-path-segment"><a href="/calispac/digicampipe/tree/lid_cdd_analysis/digicampipe/image"><span>image</span></a></span><span class="separator">/</span><strong class="final-path">cones_image.py</strong>
    </div>
  </div>


  <include-fragment class="commit-tease" src="/calispac/digicampipe/contributors/lid_cdd_analysis/digicampipe/image/cones_image.py">
    <div>
      Fetching contributors&hellip;
    </div>

    <div class="commit-tease-contributors">
      <img alt="" class="loader-loading float-left" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32-EAF2F5.gif" width="16" />
      <span class="loader-error">Cannot retrieve contributors at this time</span>
    </div>
</include-fragment>

  <div class="file">
    <div class="file-header">
  <div class="file-actions">

    <div class="BtnGroup">
      <a href="/calispac/digicampipe/raw/lid_cdd_analysis/digicampipe/image/cones_image.py" class="btn btn-sm BtnGroup-item" id="raw-url">Raw</a>
        <a href="/calispac/digicampipe/blame/lid_cdd_analysis/digicampipe/image/cones_image.py" class="btn btn-sm js-update-url-with-hash BtnGroup-item" data-hotkey="b">Blame</a>
      <a href="/calispac/digicampipe/commits/lid_cdd_analysis/digicampipe/image/cones_image.py" class="btn btn-sm BtnGroup-item" rel="nofollow">History</a>
    </div>


        <button type="button" class="btn-octicon disabled tooltipped tooltipped-nw"
          aria-label="You must be signed in to make or propose changes">
          <svg aria-hidden="true" class="octicon octicon-pencil" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"/></svg>
        </button>
        <button type="button" class="btn-octicon btn-octicon-danger disabled tooltipped tooltipped-nw"
          aria-label="You must be signed in to make or propose changes">
          <svg aria-hidden="true" class="octicon octicon-trashcan" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M11 2H9c0-.55-.45-1-1-1H5c-.55 0-1 .45-1 1H2c-.55 0-1 .45-1 1v1c0 .55.45 1 1 1v9c0 .55.45 1 1 1h7c.55 0 1-.45 1-1V5c.55 0 1-.45 1-1V3c0-.55-.45-1-1-1zm-1 12H3V5h1v8h1V5h1v8h1V5h1v8h1V5h1v9zm1-10H2V3h9v1z"/></svg>
        </button>
  </div>

  <div class="file-info">
      678 lines (666 sloc)
      <span class="file-info-divider"></span>
    38.2 KB
  </div>
</div>

    

  <div itemprop="text" class="blob-wrapper data type-python">
      <table class="highlight tab-size js-file-line-container" data-tab-size="8">
      <tr>
        <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
        <td id="LC1" class="blob-code blob-code-inner js-file-line"><span class="pl-k">from</span> digicampipe.utils <span class="pl-k">import</span> geometry</td>
      </tr>
      <tr>
        <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
        <td id="LC2" class="blob-code blob-code-inner js-file-line"><span class="pl-k">from</span> digicampipe.image.kernels <span class="pl-k">import</span> <span class="pl-k">*</span></td>
      </tr>
      <tr>
        <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
        <td id="LC3" class="blob-code blob-code-inner js-file-line"><span class="pl-k">from</span> digicampipe.image.utils <span class="pl-k">import</span> <span class="pl-k">*</span></td>
      </tr>
      <tr>
        <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
        <td id="LC4" class="blob-code blob-code-inner js-file-line"><span class="pl-k">from</span> cts_core.camera <span class="pl-k">import</span> Camera</td>
      </tr>
      <tr>
        <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
        <td id="LC5" class="blob-code blob-code-inner js-file-line"><span class="pl-k">from</span> astropy <span class="pl-k">import</span> units <span class="pl-k">as</span> u</td>
      </tr>
      <tr>
        <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
        <td id="LC6" class="blob-code blob-code-inner js-file-line"><span class="pl-k">from</span> matplotlib.patches <span class="pl-k">import</span> Circle, Arrow</td>
      </tr>
      <tr>
        <td id="L7" class="blob-num js-line-number" data-line-number="7"></td>
        <td id="LC7" class="blob-code blob-code-inner js-file-line"><span class="pl-k">import</span> numpy <span class="pl-k">as</span> np</td>
      </tr>
      <tr>
        <td id="L8" class="blob-num js-line-number" data-line-number="8"></td>
        <td id="LC8" class="blob-code blob-code-inner js-file-line"><span class="pl-k">import</span> os</td>
      </tr>
      <tr>
        <td id="L9" class="blob-num js-line-number" data-line-number="9"></td>
        <td id="LC9" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L10" class="blob-num js-line-number" data-line-number="10"></td>
        <td id="LC10" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L11" class="blob-num js-line-number" data-line-number="11"></td>
        <td id="LC11" class="blob-code blob-code-inner js-file-line"><span class="pl-k">class</span> <span class="pl-en">ConesImage</span>(<span class="pl-c1">object</span>):</td>
      </tr>
      <tr>
        <td id="L12" class="blob-num js-line-number" data-line-number="12"></td>
        <td id="LC12" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__init__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">image</span>, <span class="pl-smi">image_cone</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">output_dir</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>./data/test/cone_simu/<span class="pl-pds">&#39;</span></span>):</td>
      </tr>
      <tr>
        <td id="L13" class="blob-num js-line-number" data-line-number="13"></td>
        <td id="LC13" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L14" class="blob-num js-line-number" data-line-number="14"></td>
        <td id="LC14" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        constructor of a ConesImage object.</span></td>
      </tr>
      <tr>
        <td id="L15" class="blob-num js-line-number" data-line-number="15"></td>
        <td id="LC15" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param image: fit filename or numpy array containing the lid CCD image.</span></td>
      </tr>
      <tr>
        <td id="L16" class="blob-num js-line-number" data-line-number="16"></td>
        <td id="LC16" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        If set to &#39;test&#39;, a test image is created.</span></td>
      </tr>
      <tr>
        <td id="L17" class="blob-num js-line-number" data-line-number="17"></td>
        <td id="LC17" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param image_cone: optional fit filename of the cone image. the fit file is created calling get_cone()</span></td>
      </tr>
      <tr>
        <td id="L18" class="blob-num js-line-number" data-line-number="18"></td>
        <td id="LC18" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param output_dir: optional directory where to put the original lid CCD image in case of test</span></td>
      </tr>
      <tr>
        <td id="L19" class="blob-num js-line-number" data-line-number="19"></td>
        <td id="LC19" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L20" class="blob-num js-line-number" data-line-number="20"></td>
        <td id="LC20" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.filename <span class="pl-k">=</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L21" class="blob-num js-line-number" data-line-number="21"></td>
        <td id="LC21" class="blob-code blob-code-inner js-file-line">        digicam_config_file <span class="pl-k">=</span> <span class="pl-s"><span class="pl-pds">&#39;</span>/usr/src/cts/config/camera_config.cfg<span class="pl-pds">&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L22" class="blob-num js-line-number" data-line-number="22"></td>
        <td id="LC22" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> Camera and Geometry objects (mapping, pixel, patch + x,y coordinates pixels)</span></td>
      </tr>
      <tr>
        <td id="L23" class="blob-num js-line-number" data-line-number="23"></td>
        <td id="LC23" class="blob-code blob-code-inner js-file-line">        digicam <span class="pl-k">=</span> Camera(<span class="pl-v">_config_file</span><span class="pl-k">=</span>digicam_config_file)</td>
      </tr>
      <tr>
        <td id="L24" class="blob-num js-line-number" data-line-number="24"></td>
        <td id="LC24" class="blob-code blob-code-inner js-file-line">        digicam_geometry <span class="pl-k">=</span> geometry.generate_geometry_from_camera(<span class="pl-v">camera</span><span class="pl-k">=</span>digicam)</td>
      </tr>
      <tr>
        <td id="L25" class="blob-num js-line-number" data-line-number="25"></td>
        <td id="LC25" class="blob-code blob-code-inner js-file-line">        pixels_pos_mm <span class="pl-k">=</span> np.array([digicam_geometry.pix_x.to(u.mm), digicam_geometry.pix_y.to(u.mm)]).transpose()</td>
      </tr>
      <tr>
        <td id="L26" class="blob-num js-line-number" data-line-number="26"></td>
        <td id="LC26" class="blob-code blob-code-inner js-file-line">        pixels_pos_mm <span class="pl-k">=</span> pixels_pos_mm.dot(np.array([[<span class="pl-c1">0</span>, <span class="pl-k">-</span><span class="pl-c1">1</span>], [<span class="pl-c1">1</span>, <span class="pl-c1">0</span>]]))</td>
      </tr>
      <tr>
        <td id="L27" class="blob-num js-line-number" data-line-number="27"></td>
        <td id="LC27" class="blob-code blob-code-inner js-file-line">        pixels_v1 <span class="pl-k">=</span> pixels_pos_mm[digicam_geometry.neighbors[<span class="pl-c1">0</span>][<span class="pl-c1">1</span>], :] <span class="pl-k">-</span> pixels_pos_mm[<span class="pl-c1">0</span>, :]</td>
      </tr>
      <tr>
        <td id="L28" class="blob-num js-line-number" data-line-number="28"></td>
        <td id="LC28" class="blob-code blob-code-inner js-file-line">        pixels_v2 <span class="pl-k">=</span> pixels_pos_mm[digicam_geometry.neighbors[<span class="pl-c1">0</span>][<span class="pl-c1">0</span>], :] <span class="pl-k">-</span> pixels_pos_mm[<span class="pl-c1">0</span>, :]</td>
      </tr>
      <tr>
        <td id="L29" class="blob-num js-line-number" data-line-number="29"></td>
        <td id="LC29" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> plot_points(self.pixels_pos_mm[:, 0], self.pixels_pos_mm[:, 1])</span></td>
      </tr>
      <tr>
        <td id="L30" class="blob-num js-line-number" data-line-number="30"></td>
        <td id="LC30" class="blob-code blob-code-inner js-file-line">        index_to_pos <span class="pl-k">=</span> np.array([pixels_v1, pixels_v2]).transpose()</td>
      </tr>
      <tr>
        <td id="L31" class="blob-num js-line-number" data-line-number="31"></td>
        <td id="LC31" class="blob-code blob-code-inner js-file-line">        relative_pos <span class="pl-k">=</span> (pixels_pos_mm <span class="pl-k">-</span> pixels_pos_mm[<span class="pl-c1">0</span>, :]).transpose()</td>
      </tr>
      <tr>
        <td id="L32" class="blob-num js-line-number" data-line-number="32"></td>
        <td id="LC32" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.pixels_nvs <span class="pl-k">=</span> np.linalg.pinv(index_to_pos).dot(relative_pos)</td>
      </tr>
      <tr>
        <td id="L33" class="blob-num js-line-number" data-line-number="33"></td>
        <td id="LC33" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.pixels_nvs <span class="pl-k">-=</span> np.round(np.mean(<span class="pl-c1">self</span>.pixels_nvs, <span class="pl-v">axis</span><span class="pl-k">=</span><span class="pl-c1">1</span>)).reshape(<span class="pl-c1">2</span>, <span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L34" class="blob-num js-line-number" data-line-number="34"></td>
        <td id="LC34" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.pixels_nvs <span class="pl-k">=</span> np.round(<span class="pl-c1">self</span>.pixels_nvs).astype(<span class="pl-c1">int</span>)</td>
      </tr>
      <tr>
        <td id="L35" class="blob-num js-line-number" data-line-number="35"></td>
        <td id="LC35" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">type</span>(image) <span class="pl-k">is</span> <span class="pl-c1">str</span>:</td>
      </tr>
      <tr>
        <td id="L36" class="blob-num js-line-number" data-line-number="36"></td>
        <td id="LC36" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> image <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>test<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L37" class="blob-num js-line-number" data-line-number="37"></td>
        <td id="LC37" class="blob-code blob-code-inner js-file-line">                angle_rot <span class="pl-k">=</span> <span class="pl-c1">3.5</span> <span class="pl-k">/</span> <span class="pl-c1">180</span> <span class="pl-k">*</span> np.pi</td>
      </tr>
      <tr>
        <td id="L38" class="blob-num js-line-number" data-line-number="38"></td>
        <td id="LC38" class="blob-code blob-code-inner js-file-line">                offset <span class="pl-k">=</span> np.array((<span class="pl-c1">25</span>, <span class="pl-c1">5</span>))</td>
      </tr>
      <tr>
        <td id="L39" class="blob-num js-line-number" data-line-number="39"></td>
        <td id="LC39" class="blob-code blob-code-inner js-file-line">                image <span class="pl-k">=</span> np.zeros((<span class="pl-c1">2472</span>, <span class="pl-c1">3296</span>))</td>
      </tr>
      <tr>
        <td id="L40" class="blob-num js-line-number" data-line-number="40"></td>
        <td id="LC40" class="blob-code blob-code-inner js-file-line">                center <span class="pl-k">=</span> (np.array(image.shape[::<span class="pl-k">-</span><span class="pl-c1">1</span>]) <span class="pl-k">-</span> <span class="pl-c1">1</span>) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L41" class="blob-num js-line-number" data-line-number="41"></td>
        <td id="LC41" class="blob-code blob-code-inner js-file-line">                r1 <span class="pl-k">=</span> <span class="pl-c1">38.3</span> <span class="pl-k">*</span> np.array((np.cos(angle_rot), np.sin(angle_rot)))</td>
      </tr>
      <tr>
        <td id="L42" class="blob-num js-line-number" data-line-number="42"></td>
        <td id="LC42" class="blob-code blob-code-inner js-file-line">                r2 <span class="pl-k">=</span> <span class="pl-c1">38.3</span> <span class="pl-k">*</span> np.array((np.cos(np.pi <span class="pl-k">/</span> <span class="pl-c1">3</span> <span class="pl-k">+</span> angle_rot), np.sin(np.pi <span class="pl-k">/</span> <span class="pl-c1">3</span> <span class="pl-k">+</span> angle_rot)))</td>
      </tr>
      <tr>
        <td id="L43" class="blob-num js-line-number" data-line-number="43"></td>
        <td id="LC43" class="blob-code blob-code-inner js-file-line">                v1_lattice <span class="pl-k">=</span> <span class="pl-c1">2</span> <span class="pl-k">*</span> r1 <span class="pl-k">-</span> r2</td>
      </tr>
      <tr>
        <td id="L44" class="blob-num js-line-number" data-line-number="44"></td>
        <td id="LC44" class="blob-code blob-code-inner js-file-line">                v2_lattice <span class="pl-k">=</span> r1 <span class="pl-k">+</span> r2</td>
      </tr>
      <tr>
        <td id="L45" class="blob-num js-line-number" data-line-number="45"></td>
        <td id="LC45" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>test lattice with v1=<span class="pl-pds">&#39;</span></span>, v1_lattice, <span class="pl-s"><span class="pl-pds">&#39;</span>v2=<span class="pl-pds">&#39;</span></span>, v2_lattice, <span class="pl-s"><span class="pl-pds">&#39;</span>offset=<span class="pl-pds">&#39;</span></span>, offset)</td>
      </tr>
      <tr>
        <td id="L46" class="blob-num js-line-number" data-line-number="46"></td>
        <td id="LC46" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">for</span> nv1, nv2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.pixels_nvs.transpose():</td>
      </tr>
      <tr>
        <td id="L47" class="blob-num js-line-number" data-line-number="47"></td>
        <td id="LC47" class="blob-code blob-code-inner js-file-line">                    <span class="pl-k">for</span> i <span class="pl-k">in</span> <span class="pl-c1">range</span>(<span class="pl-c1">10</span>, <span class="pl-k">-</span><span class="pl-c1">1</span>, <span class="pl-k">-</span><span class="pl-c1">1</span>):</td>
      </tr>
      <tr>
        <td id="L48" class="blob-num js-line-number" data-line-number="48"></td>
        <td id="LC48" class="blob-code blob-code-inner js-file-line">                        image <span class="pl-k">=</span> set_hexagon(image, center <span class="pl-k">+</span> offset <span class="pl-k">+</span> nv1 <span class="pl-k">*</span> v1_lattice <span class="pl-k">+</span> nv2 <span class="pl-k">*</span> v2_lattice,</td>
      </tr>
      <tr>
        <td id="L49" class="blob-num js-line-number" data-line-number="49"></td>
        <td id="LC49" class="blob-code blob-code-inner js-file-line">                                            <span class="pl-v">r1</span><span class="pl-k">=</span>(i<span class="pl-k">+</span><span class="pl-c1">7</span>)<span class="pl-k">/</span><span class="pl-c1">20</span> <span class="pl-k">*</span> r1 , <span class="pl-v">r2</span><span class="pl-k">=</span>(i<span class="pl-k">+</span><span class="pl-c1">7</span>)<span class="pl-k">/</span><span class="pl-c1">20</span> <span class="pl-k">*</span> r2, <span class="pl-v">value</span><span class="pl-k">=</span><span class="pl-c1">1</span><span class="pl-k">-</span>i<span class="pl-k">/</span><span class="pl-c1">10</span>)</td>
      </tr>
      <tr>
        <td id="L50" class="blob-num js-line-number" data-line-number="50"></td>
        <td id="LC50" class="blob-code blob-code-inner js-file-line">                    image <span class="pl-k">=</span> set_hexagon(image, center <span class="pl-k">+</span> offset <span class="pl-k">+</span> nv1 <span class="pl-k">*</span> v1_lattice <span class="pl-k">+</span> nv2 <span class="pl-k">*</span> v2_lattice,</td>
      </tr>
      <tr>
        <td id="L51" class="blob-num js-line-number" data-line-number="51"></td>
        <td id="LC51" class="blob-code blob-code-inner js-file-line">                                        <span class="pl-v">r1</span><span class="pl-k">=</span><span class="pl-c1">6</span><span class="pl-k">/</span><span class="pl-c1">20</span> <span class="pl-k">*</span> r1, <span class="pl-v">r2</span><span class="pl-k">=</span><span class="pl-c1">6</span><span class="pl-k">/</span><span class="pl-c1">20</span> <span class="pl-k">*</span> r2, <span class="pl-v">value</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L52" class="blob-num js-line-number" data-line-number="52"></td>
        <td id="LC52" class="blob-code blob-code-inner js-file-line">                image <span class="pl-k">+=</span> <span class="pl-c1">0</span>. <span class="pl-k">*</span> np.random.randn(image.shape[<span class="pl-c1">0</span>],image.shape[<span class="pl-c1">1</span>])</td>
      </tr>
      <tr>
        <td id="L53" class="blob-num js-line-number" data-line-number="53"></td>
        <td id="LC53" class="blob-code blob-code-inner js-file-line">                image[<span class="pl-c1">0</span>,<span class="pl-c1">0</span>]<span class="pl-k">=</span><span class="pl-c1">10000</span></td>
      </tr>
      <tr>
        <td id="L54" class="blob-num js-line-number" data-line-number="54"></td>
        <td id="LC54" class="blob-code blob-code-inner js-file-line">                fig <span class="pl-k">=</span> plt.figure()</td>
      </tr>
      <tr>
        <td id="L55" class="blob-num js-line-number" data-line-number="55"></td>
        <td id="LC55" class="blob-code blob-code-inner js-file-line">                ax <span class="pl-k">=</span> plt.gca()</td>
      </tr>
      <tr>
        <td id="L56" class="blob-num js-line-number" data-line-number="56"></td>
        <td id="LC56" class="blob-code blob-code-inner js-file-line">                plt.imshow(image, <span class="pl-v">cmap</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>gray<span class="pl-pds">&#39;</span></span>, <span class="pl-v">vmin</span><span class="pl-k">=</span><span class="pl-c1">0</span>, <span class="pl-v">vmax</span><span class="pl-k">=</span><span class="pl-c1">10</span>)</td>
      </tr>
      <tr>
        <td id="L57" class="blob-num js-line-number" data-line-number="57"></td>
        <td id="LC57" class="blob-code blob-code-inner js-file-line">                plt.grid(<span class="pl-c1">None</span>)</td>
      </tr>
      <tr>
        <td id="L58" class="blob-num js-line-number" data-line-number="58"></td>
        <td id="LC58" class="blob-code blob-code-inner js-file-line">                plt.axis(<span class="pl-s"><span class="pl-pds">&#39;</span>off<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L59" class="blob-num js-line-number" data-line-number="59"></td>
        <td id="LC59" class="blob-code blob-code-inner js-file-line">                ax.get_xaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L60" class="blob-num js-line-number" data-line-number="60"></td>
        <td id="LC60" class="blob-code blob-code-inner js-file-line">                ax.get_yaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L61" class="blob-num js-line-number" data-line-number="61"></td>
        <td id="LC61" class="blob-code blob-code-inner js-file-line">                output_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>cones-original.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L62" class="blob-num js-line-number" data-line-number="62"></td>
        <td id="LC62" class="blob-code blob-code-inner js-file-line">                plt.savefig(output_filename, <span class="pl-v">bbox_inches</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>tight<span class="pl-pds">&#39;</span></span>, <span class="pl-v">pad_inches</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L63" class="blob-num js-line-number" data-line-number="63"></td>
        <td id="LC63" class="blob-code blob-code-inner js-file-line">                plt.close(fig)</td>
      </tr>
      <tr>
        <td id="L64" class="blob-num js-line-number" data-line-number="64"></td>
        <td id="LC64" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">print</span>(output_filename, <span class="pl-s"><span class="pl-pds">&#39;</span>saved.<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L65" class="blob-num js-line-number" data-line-number="65"></td>
        <td id="LC65" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L66" class="blob-num js-line-number" data-line-number="66"></td>
        <td id="LC66" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">self</span>.filename <span class="pl-k">=</span> image</td>
      </tr>
      <tr>
        <td id="L67" class="blob-num js-line-number" data-line-number="67"></td>
        <td id="LC67" class="blob-code blob-code-inner js-file-line">                image <span class="pl-k">=</span> fits.open(image)[<span class="pl-c1">0</span>].data</td>
      </tr>
      <tr>
        <td id="L68" class="blob-num js-line-number" data-line-number="68"></td>
        <td id="LC68" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">type</span>(image) <span class="pl-k">is</span> <span class="pl-k">not</span> np.ndarray:</td>
      </tr>
      <tr>
        <td id="L69" class="blob-num js-line-number" data-line-number="69"></td>
        <td id="LC69" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">raise</span> <span class="pl-c1">AttributeError</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>image must be a filename or a numpy.ndarray<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L70" class="blob-num js-line-number" data-line-number="70"></td>
        <td id="LC70" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> high pass filter</span></td>
      </tr>
      <tr>
        <td id="L71" class="blob-num js-line-number" data-line-number="71"></td>
        <td id="LC71" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.image_cones <span class="pl-k">=</span> signal.convolve2d(image, high_pass_filter_77, <span class="pl-v">mode</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>same<span class="pl-pds">&#39;</span></span>, <span class="pl-v">boundary</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>symm<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L72" class="blob-num js-line-number" data-line-number="72"></td>
        <td id="LC72" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> low_pass_filter</span></td>
      </tr>
      <tr>
        <td id="L73" class="blob-num js-line-number" data-line-number="73"></td>
        <td id="LC73" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.image_cones <span class="pl-k">=</span> signal.convolve2d(<span class="pl-c1">self</span>.image_cones, gauss(<span class="pl-c1">3</span>, (<span class="pl-c1">15</span>, <span class="pl-c1">15</span>)), <span class="pl-v">mode</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>same<span class="pl-pds">&#39;</span></span>, <span class="pl-v">boundary</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>symm<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L74" class="blob-num js-line-number" data-line-number="74"></td>
        <td id="LC74" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> thresholds</span></td>
      </tr>
      <tr>
        <td id="L75" class="blob-num js-line-number" data-line-number="75"></td>
        <td id="LC75" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.image_cones[<span class="pl-c1">self</span>.image_cones <span class="pl-k">&lt;</span> <span class="pl-c1">0</span>] <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L76" class="blob-num js-line-number" data-line-number="76"></td>
        <td id="LC76" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.image_cones[<span class="pl-c1">self</span>.image_cones <span class="pl-k">&gt;</span> np.std(<span class="pl-c1">self</span>.image_cones)] <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L77" class="blob-num js-line-number" data-line-number="77"></td>
        <td id="LC77" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.image_cones <span class="pl-k">-=</span> np.mean(<span class="pl-c1">self</span>.image_cones)</td>
      </tr>
      <tr>
        <td id="L78" class="blob-num js-line-number" data-line-number="78"></td>
        <td id="LC78" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> fft</span></td>
      </tr>
      <tr>
        <td id="L79" class="blob-num js-line-number" data-line-number="79"></td>
        <td id="LC79" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.fft_image_cones <span class="pl-k">=</span> np.fft.fft2(<span class="pl-c1">self</span>.image_cones)</td>
      </tr>
      <tr>
        <td id="L80" class="blob-num js-line-number" data-line-number="80"></td>
        <td id="LC80" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.center_fitted <span class="pl-k">=</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L81" class="blob-num js-line-number" data-line-number="81"></td>
        <td id="LC81" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> filled by get_cones_separation_reciprocal():</span></td>
      </tr>
      <tr>
        <td id="L82" class="blob-num js-line-number" data-line-number="82"></td>
        <td id="LC82" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ks <span class="pl-k">=</span> <span class="pl-c1">None</span> <span class="pl-c"><span class="pl-c">#</span> distance between fft peaks</span></td>
      </tr>
      <tr>
        <td id="L83" class="blob-num js-line-number" data-line-number="83"></td>
        <td id="LC83" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.v1_lattice <span class="pl-k">=</span> <span class="pl-c1">None</span> <span class="pl-c"><span class="pl-c">#</span> distance between 2 hexagons</span></td>
      </tr>
      <tr>
        <td id="L84" class="blob-num js-line-number" data-line-number="84"></td>
        <td id="LC84" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.v2_lattice <span class="pl-k">=</span> <span class="pl-c1">None</span> <span class="pl-c"><span class="pl-c">#</span> distance between 2 hexagons</span></td>
      </tr>
      <tr>
        <td id="L85" class="blob-num js-line-number" data-line-number="85"></td>
        <td id="LC85" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.r1 <span class="pl-k">=</span> <span class="pl-c1">None</span> <span class="pl-c"><span class="pl-c">#</span> radius of the hexagon pixel</span></td>
      </tr>
      <tr>
        <td id="L86" class="blob-num js-line-number" data-line-number="86"></td>
        <td id="LC86" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.r2 <span class="pl-k">=</span> <span class="pl-c1">None</span> <span class="pl-c"><span class="pl-c">#</span> radius of the hexagon (after r1 in anticlockwise)</span></td>
      </tr>
      <tr>
        <td id="L87" class="blob-num js-line-number" data-line-number="87"></td>
        <td id="LC87" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.r3 <span class="pl-k">=</span> <span class="pl-c1">None</span> <span class="pl-c"><span class="pl-c">#</span> radius of the hexagon (after r2 in anticlockwise)</span></td>
      </tr>
      <tr>
        <td id="L88" class="blob-num js-line-number" data-line-number="88"></td>
        <td id="LC88" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> individual cone image, call get_cone() function to compute it</span></td>
      </tr>
      <tr>
        <td id="L89" class="blob-num js-line-number" data-line-number="89"></td>
        <td id="LC89" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> image_cone <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L90" class="blob-num js-line-number" data-line-number="90"></td>
        <td id="LC90" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.image_cone <span class="pl-k">=</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L91" class="blob-num js-line-number" data-line-number="91"></td>
        <td id="LC91" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L92" class="blob-num js-line-number" data-line-number="92"></td>
        <td id="LC92" class="blob-code blob-code-inner js-file-line">            hdu<span class="pl-k">=</span>fits.open(image_cone)[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L93" class="blob-num js-line-number" data-line-number="93"></td>
        <td id="LC93" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.image_cone <span class="pl-k">=</span> hdu.data</td>
      </tr>
      <tr>
        <td id="L94" class="blob-num js-line-number" data-line-number="94"></td>
        <td id="LC94" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.center_fitted <span class="pl-k">=</span> np.array((np.real(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>center<span class="pl-pds">&#39;</span></span>]),np.imag(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>center<span class="pl-pds">&#39;</span></span>])))</td>
      </tr>
      <tr>
        <td id="L95" class="blob-num js-line-number" data-line-number="95"></td>
        <td id="LC95" class="blob-code blob-code-inner js-file-line">            center <span class="pl-k">=</span> (np.array(<span class="pl-c1">self</span>.image_cones.shape[::<span class="pl-k">-</span><span class="pl-c1">1</span>]) <span class="pl-k">-</span> <span class="pl-c1">1</span>) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L96" class="blob-num js-line-number" data-line-number="96"></td>
        <td id="LC96" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>center of loaded cone at<span class="pl-pds">&#39;</span></span>, <span class="pl-c1">self</span>.center_fitted, <span class="pl-s"><span class="pl-pds">&#39;</span>,<span class="pl-pds">&#39;</span></span>, <span class="pl-c1">self</span>.center_fitted <span class="pl-k">-</span> center, <span class="pl-s"><span class="pl-pds">&#39;</span>from center<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L97" class="blob-num js-line-number" data-line-number="97"></td>
        <td id="LC97" class="blob-code blob-code-inner js-file-line">            ks1 <span class="pl-k">=</span> np.array((np.real(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>ks1<span class="pl-pds">&#39;</span></span>]),np.imag(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>ks1<span class="pl-pds">&#39;</span></span>])))</td>
      </tr>
      <tr>
        <td id="L98" class="blob-num js-line-number" data-line-number="98"></td>
        <td id="LC98" class="blob-code blob-code-inner js-file-line">            ks2 <span class="pl-k">=</span> np.array((np.real(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>ks2<span class="pl-pds">&#39;</span></span>]), np.imag(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>ks2<span class="pl-pds">&#39;</span></span>])))</td>
      </tr>
      <tr>
        <td id="L99" class="blob-num js-line-number" data-line-number="99"></td>
        <td id="LC99" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.ks <span class="pl-k">=</span> np.array((ks1, ks2))</td>
      </tr>
      <tr>
        <td id="L100" class="blob-num js-line-number" data-line-number="100"></td>
        <td id="LC100" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.v1_lattice <span class="pl-k">=</span> np.array((np.real(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>v1<span class="pl-pds">&#39;</span></span>]),np.imag(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>v1<span class="pl-pds">&#39;</span></span>])))</td>
      </tr>
      <tr>
        <td id="L101" class="blob-num js-line-number" data-line-number="101"></td>
        <td id="LC101" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.v2_lattice <span class="pl-k">=</span> np.array((np.real(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>v2<span class="pl-pds">&#39;</span></span>]),np.imag(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>v2<span class="pl-pds">&#39;</span></span>])))</td>
      </tr>
      <tr>
        <td id="L102" class="blob-num js-line-number" data-line-number="102"></td>
        <td id="LC102" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.r1 <span class="pl-k">=</span> np.array((np.real(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>r1<span class="pl-pds">&#39;</span></span>]),np.imag(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>r1<span class="pl-pds">&#39;</span></span>])))</td>
      </tr>
      <tr>
        <td id="L103" class="blob-num js-line-number" data-line-number="103"></td>
        <td id="LC103" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.r2 <span class="pl-k">=</span> np.array((np.real(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>r2<span class="pl-pds">&#39;</span></span>]),np.imag(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>r2<span class="pl-pds">&#39;</span></span>])))</td>
      </tr>
      <tr>
        <td id="L104" class="blob-num js-line-number" data-line-number="104"></td>
        <td id="LC104" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.r3 <span class="pl-k">=</span> np.array((np.real(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>r3<span class="pl-pds">&#39;</span></span>]),np.imag(hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>r3<span class="pl-pds">&#39;</span></span>])))</td>
      </tr>
      <tr>
        <td id="L105" class="blob-num js-line-number" data-line-number="105"></td>
        <td id="LC105" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L106" class="blob-num js-line-number" data-line-number="106"></td>
        <td id="LC106" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">plot_cones</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">radius_mask</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">output_dir</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L107" class="blob-num js-line-number" data-line-number="107"></td>
        <td id="LC107" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L108" class="blob-num js-line-number" data-line-number="108"></td>
        <td id="LC108" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        If radius_mask is None, plot the lid CCD image after filtering, otherwise plot the cones contents of the image.</span></td>
      </tr>
      <tr>
        <td id="L109" class="blob-num js-line-number" data-line-number="109"></td>
        <td id="LC109" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        The cones contents is obtained applying a mask of radius radius_mask around each peak of the FFT and doing the</span></td>
      </tr>
      <tr>
        <td id="L110" class="blob-num js-line-number" data-line-number="110"></td>
        <td id="LC110" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        inverse FFT transformation.</span></td>
      </tr>
      <tr>
        <td id="L111" class="blob-num js-line-number" data-line-number="111"></td>
        <td id="LC111" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param radius_mask: optional radius of the mask used for each peak in the FFT</span></td>
      </tr>
      <tr>
        <td id="L112" class="blob-num js-line-number" data-line-number="112"></td>
        <td id="LC112" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param output_dir: optional directory where to put the resulting image. If None (default) the image is displayed</span></td>
      </tr>
      <tr>
        <td id="L113" class="blob-num js-line-number" data-line-number="113"></td>
        <td id="LC113" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        instead of being saved to a file.</span></td>
      </tr>
      <tr>
        <td id="L114" class="blob-num js-line-number" data-line-number="114"></td>
        <td id="LC114" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L115" class="blob-num js-line-number" data-line-number="115"></td>
        <td id="LC115" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L116" class="blob-num js-line-number" data-line-number="116"></td>
        <td id="LC116" class="blob-code blob-code-inner js-file-line">            plt.ioff()</td>
      </tr>
      <tr>
        <td id="L117" class="blob-num js-line-number" data-line-number="117"></td>
        <td id="LC117" class="blob-code blob-code-inner js-file-line">        fig <span class="pl-k">=</span> plt.figure()</td>
      </tr>
      <tr>
        <td id="L118" class="blob-num js-line-number" data-line-number="118"></td>
        <td id="LC118" class="blob-code blob-code-inner js-file-line">        ax <span class="pl-k">=</span> plt.gca()</td>
      </tr>
      <tr>
        <td id="L119" class="blob-num js-line-number" data-line-number="119"></td>
        <td id="LC119" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">type</span>(<span class="pl-c1">self</span>.image_cones) <span class="pl-k">is</span> <span class="pl-k">not</span> np.ndarray:</td>
      </tr>
      <tr>
        <td id="L120" class="blob-num js-line-number" data-line-number="120"></td>
        <td id="LC120" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">raise</span> <span class="pl-c1">AttributeError</span>([<span class="pl-c1">self</span>.filename, <span class="pl-s"><span class="pl-pds">&#39;</span> must be a fit file<span class="pl-pds">&#39;</span></span>])</td>
      </tr>
      <tr>
        <td id="L121" class="blob-num js-line-number" data-line-number="121"></td>
        <td id="LC121" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> radius_mask <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L122" class="blob-num js-line-number" data-line-number="122"></td>
        <td id="LC122" class="blob-code blob-code-inner js-file-line">            mask <span class="pl-k">=</span> <span class="pl-c1">self</span>.get_fft_mask(<span class="pl-v">radius</span><span class="pl-k">=</span>radius_mask)</td>
      </tr>
      <tr>
        <td id="L123" class="blob-num js-line-number" data-line-number="123"></td>
        <td id="LC123" class="blob-code blob-code-inner js-file-line">            image_cones <span class="pl-k">=</span> np.real(np.fft.ifft2(<span class="pl-c1">self</span>.fft_image_cones <span class="pl-k">*</span> mask))</td>
      </tr>
      <tr>
        <td id="L124" class="blob-num js-line-number" data-line-number="124"></td>
        <td id="LC124" class="blob-code blob-code-inner js-file-line">            image_cones <span class="pl-k">-=</span> np.min(image_cones)</td>
      </tr>
      <tr>
        <td id="L125" class="blob-num js-line-number" data-line-number="125"></td>
        <td id="LC125" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L126" class="blob-num js-line-number" data-line-number="126"></td>
        <td id="LC126" class="blob-code blob-code-inner js-file-line">            image_cones<span class="pl-k">=</span><span class="pl-c1">self</span>.image_cones</td>
      </tr>
      <tr>
        <td id="L127" class="blob-num js-line-number" data-line-number="127"></td>
        <td id="LC127" class="blob-code blob-code-inner js-file-line">        plt.imshow(image_cones, <span class="pl-v">cmap</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>gray<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L128" class="blob-num js-line-number" data-line-number="128"></td>
        <td id="LC128" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> radius_mask <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L129" class="blob-num js-line-number" data-line-number="129"></td>
        <td id="LC129" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> <span class="pl-c1">self</span>.center_fitted <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L130" class="blob-num js-line-number" data-line-number="130"></td>
        <td id="LC130" class="blob-code blob-code-inner js-file-line">                center <span class="pl-k">=</span> (np.array(<span class="pl-c1">self</span>.image_cones.shape)[::<span class="pl-k">-</span><span class="pl-c1">1</span>] <span class="pl-k">-</span> <span class="pl-c1">1</span>) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L131" class="blob-num js-line-number" data-line-number="131"></td>
        <td id="LC131" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L132" class="blob-num js-line-number" data-line-number="132"></td>
        <td id="LC132" class="blob-code blob-code-inner js-file-line">                center <span class="pl-k">=</span> <span class="pl-c1">self</span>.center_fitted</td>
      </tr>
      <tr>
        <td id="L133" class="blob-num js-line-number" data-line-number="133"></td>
        <td id="LC133" class="blob-code blob-code-inner js-file-line">            v1_points <span class="pl-k">=</span> np.array((np.zeros((<span class="pl-c1">2</span>,)), <span class="pl-c1">self</span>.v1_lattice)) <span class="pl-k">+</span> center</td>
      </tr>
      <tr>
        <td id="L134" class="blob-num js-line-number" data-line-number="134"></td>
        <td id="LC134" class="blob-code blob-code-inner js-file-line">            v2_points <span class="pl-k">=</span> np.array((np.zeros((<span class="pl-c1">2</span>,)), <span class="pl-c1">self</span>.v2_lattice)) <span class="pl-k">+</span> center</td>
      </tr>
      <tr>
        <td id="L135" class="blob-num js-line-number" data-line-number="135"></td>
        <td id="LC135" class="blob-code blob-code-inner js-file-line">            r1_points <span class="pl-k">=</span> np.array((np.zeros((<span class="pl-c1">2</span>,)), <span class="pl-c1">self</span>.r1)) <span class="pl-k">+</span> center</td>
      </tr>
      <tr>
        <td id="L136" class="blob-num js-line-number" data-line-number="136"></td>
        <td id="LC136" class="blob-code blob-code-inner js-file-line">            r2_points <span class="pl-k">=</span> np.array((np.zeros((<span class="pl-c1">2</span>,)), <span class="pl-c1">self</span>.r2)) <span class="pl-k">+</span> center</td>
      </tr>
      <tr>
        <td id="L137" class="blob-num js-line-number" data-line-number="137"></td>
        <td id="LC137" class="blob-code blob-code-inner js-file-line">            r3_points <span class="pl-k">=</span> np.array((np.zeros((<span class="pl-c1">2</span>,)), <span class="pl-c1">self</span>.r3)) <span class="pl-k">+</span> center</td>
      </tr>
      <tr>
        <td id="L138" class="blob-num js-line-number" data-line-number="138"></td>
        <td id="LC138" class="blob-code blob-code-inner js-file-line">            plt.plot(v1_points[:, <span class="pl-c1">0</span>], v1_points[:, <span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>b--<span class="pl-pds">&#39;</span></span>, <span class="pl-v">linewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L139" class="blob-num js-line-number" data-line-number="139"></td>
        <td id="LC139" class="blob-code blob-code-inner js-file-line">            plt.plot(v2_points[:, <span class="pl-c1">0</span>], v2_points[:, <span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>r--<span class="pl-pds">&#39;</span></span>, <span class="pl-v">linewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L140" class="blob-num js-line-number" data-line-number="140"></td>
        <td id="LC140" class="blob-code blob-code-inner js-file-line">            plt.plot(r1_points[:, <span class="pl-c1">0</span>], r1_points[:, <span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>b-<span class="pl-pds">&#39;</span></span>, <span class="pl-v">linewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L141" class="blob-num js-line-number" data-line-number="141"></td>
        <td id="LC141" class="blob-code blob-code-inner js-file-line">            plt.plot(r2_points[:, <span class="pl-c1">0</span>], r2_points[:, <span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>r-<span class="pl-pds">&#39;</span></span>, <span class="pl-v">linewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L142" class="blob-num js-line-number" data-line-number="142"></td>
        <td id="LC142" class="blob-code blob-code-inner js-file-line">            plt.plot(r3_points[:, <span class="pl-c1">0</span>], r3_points[:, <span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>g-<span class="pl-pds">&#39;</span></span>, <span class="pl-v">linewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L143" class="blob-num js-line-number" data-line-number="143"></td>
        <td id="LC143" class="blob-code blob-code-inner js-file-line">        plt.grid(<span class="pl-c1">None</span>)</td>
      </tr>
      <tr>
        <td id="L144" class="blob-num js-line-number" data-line-number="144"></td>
        <td id="LC144" class="blob-code blob-code-inner js-file-line">        plt.axis(<span class="pl-s"><span class="pl-pds">&#39;</span>off<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L145" class="blob-num js-line-number" data-line-number="145"></td>
        <td id="LC145" class="blob-code blob-code-inner js-file-line">        ax.get_xaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L146" class="blob-num js-line-number" data-line-number="146"></td>
        <td id="LC146" class="blob-code blob-code-inner js-file-line">        ax.get_yaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L147" class="blob-num js-line-number" data-line-number="147"></td>
        <td id="LC147" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L148" class="blob-num js-line-number" data-line-number="148"></td>
        <td id="LC148" class="blob-code blob-code-inner js-file-line">            plt.show()</td>
      </tr>
      <tr>
        <td id="L149" class="blob-num js-line-number" data-line-number="149"></td>
        <td id="LC149" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L150" class="blob-num js-line-number" data-line-number="150"></td>
        <td id="LC150" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> <span class="pl-c1">self</span>.filename <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L151" class="blob-num js-line-number" data-line-number="151"></td>
        <td id="LC151" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> radius_mask <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L152" class="blob-num js-line-number" data-line-number="152"></td>
        <td id="LC152" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> <span class="pl-c1">self</span>.filename.replace(<span class="pl-s"><span class="pl-pds">&#39;</span>.fits<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>-cones-filtered.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L153" class="blob-num js-line-number" data-line-number="153"></td>
        <td id="LC153" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L154" class="blob-num js-line-number" data-line-number="154"></td>
        <td id="LC154" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> <span class="pl-c1">self</span>.filename.replace(<span class="pl-s"><span class="pl-pds">&#39;</span>.fits<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>-cones.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L155" class="blob-num js-line-number" data-line-number="155"></td>
        <td id="LC155" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L156" class="blob-num js-line-number" data-line-number="156"></td>
        <td id="LC156" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> radius_mask <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L157" class="blob-num js-line-number" data-line-number="157"></td>
        <td id="LC157" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>cones-filtered.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L158" class="blob-num js-line-number" data-line-number="158"></td>
        <td id="LC158" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L159" class="blob-num js-line-number" data-line-number="159"></td>
        <td id="LC159" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>cones.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L160" class="blob-num js-line-number" data-line-number="160"></td>
        <td id="LC160" class="blob-code blob-code-inner js-file-line">            plt.savefig(output_filename, <span class="pl-v">bbox_inches</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>tight<span class="pl-pds">&#39;</span></span>, <span class="pl-v">pad_inches</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L161" class="blob-num js-line-number" data-line-number="161"></td>
        <td id="LC161" class="blob-code blob-code-inner js-file-line">            plt.close(fig)</td>
      </tr>
      <tr>
        <td id="L162" class="blob-num js-line-number" data-line-number="162"></td>
        <td id="LC162" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(output_filename, <span class="pl-s"><span class="pl-pds">&#39;</span>saved.<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L163" class="blob-num js-line-number" data-line-number="163"></td>
        <td id="LC163" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L164" class="blob-num js-line-number" data-line-number="164"></td>
        <td id="LC164" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">scan_cone_position</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">radius_mask</span>, <span class="pl-smi">output_dir</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">center_scan</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">rotations</span><span class="pl-k">=</span>(<span class="pl-c1">60</span>, <span class="pl-c1">300</span>)):</td>
      </tr>
      <tr>
        <td id="L165" class="blob-num js-line-number" data-line-number="165"></td>
        <td id="LC165" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L166" class="blob-num js-line-number" data-line-number="166"></td>
        <td id="LC166" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Calculate the hexagonalicity for each pixel inside a camera&#39;s pixel.</span></td>
      </tr>
      <tr>
        <td id="L167" class="blob-num js-line-number" data-line-number="167"></td>
        <td id="LC167" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param radius_mask: radius of the mask used for extracting pixels from image (see plot_cones())</span></td>
      </tr>
      <tr>
        <td id="L168" class="blob-num js-line-number" data-line-number="168"></td>
        <td id="LC168" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param output_dir: optional directory where to put the resulting image. If None (default) the image is displayed</span></td>
      </tr>
      <tr>
        <td id="L169" class="blob-num js-line-number" data-line-number="169"></td>
        <td id="LC169" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        instead of being saved to a file.</span></td>
      </tr>
      <tr>
        <td id="L170" class="blob-num js-line-number" data-line-number="170"></td>
        <td id="LC170" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param center_scan: optional position of the center of the camera&#39;s pixel.</span></td>
      </tr>
      <tr>
        <td id="L171" class="blob-num js-line-number" data-line-number="171"></td>
        <td id="LC171" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Center of image is used if None (default)</span></td>
      </tr>
      <tr>
        <td id="L172" class="blob-num js-line-number" data-line-number="172"></td>
        <td id="LC172" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param rotations: angles in degrees used for calculating hexagonalicity.</span></td>
      </tr>
      <tr>
        <td id="L173" class="blob-num js-line-number" data-line-number="173"></td>
        <td id="LC173" class="blob-code blob-code-inner js-file-line"><span class="pl-s"></span></td>
      </tr>
      <tr>
        <td id="L174" class="blob-num js-line-number" data-line-number="174"></td>
        <td id="LC174" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        remark :  output should look like</span></td>
      </tr>
      <tr>
        <td id="L175" class="blob-num js-line-number" data-line-number="175"></td>
        <td id="LC175" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        http://n.ethz.ch/~nielssi/download/4.%20Semester/AC%20II/Unterlagen/symmetry_2D_3.pdf page 22</span></td>
      </tr>
      <tr>
        <td id="L176" class="blob-num js-line-number" data-line-number="176"></td>
        <td id="LC176" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L177" class="blob-num js-line-number" data-line-number="177"></td>
        <td id="LC177" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">type</span>(radius_mask) <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">float</span>:</td>
      </tr>
      <tr>
        <td id="L178" class="blob-num js-line-number" data-line-number="178"></td>
        <td id="LC178" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">raise</span> <span class="pl-c1">AttributeError</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>radius_mask must be a float.<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L179" class="blob-num js-line-number" data-line-number="179"></td>
        <td id="LC179" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">self</span>.r1 <span class="pl-k">is</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> <span class="pl-c1">self</span>.r2 <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L180" class="blob-num js-line-number" data-line-number="180"></td>
        <td id="LC180" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">raise</span> <span class="pl-c1">AttributeError</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>camera pixel geometry must be computed prior of calling scan_cone_position().<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L181" class="blob-num js-line-number" data-line-number="181"></td>
        <td id="LC181" class="blob-code blob-code-inner js-file-line">        mask <span class="pl-k">=</span> <span class="pl-c1">self</span>.get_fft_mask(<span class="pl-v">radius</span><span class="pl-k">=</span>radius_mask)</td>
      </tr>
      <tr>
        <td id="L182" class="blob-num js-line-number" data-line-number="182"></td>
        <td id="LC182" class="blob-code blob-code-inner js-file-line">        image_cones <span class="pl-k">=</span> np.real(np.fft.ifft2(<span class="pl-c1">self</span>.fft_image_cones <span class="pl-k">*</span> mask))</td>
      </tr>
      <tr>
        <td id="L183" class="blob-num js-line-number" data-line-number="183"></td>
        <td id="LC183" class="blob-code blob-code-inner js-file-line">        center_image <span class="pl-k">=</span> (np.array(image_cones.shape[::<span class="pl-k">-</span><span class="pl-c1">1</span>]) <span class="pl-k">-</span> <span class="pl-c1">1</span>) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L184" class="blob-num js-line-number" data-line-number="184"></td>
        <td id="LC184" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> center_scan <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L185" class="blob-num js-line-number" data-line-number="185"></td>
        <td id="LC185" class="blob-code blob-code-inner js-file-line">            center_scan <span class="pl-k">=</span> center_image</td>
      </tr>
      <tr>
        <td id="L186" class="blob-num js-line-number" data-line-number="186"></td>
        <td id="LC186" class="blob-code blob-code-inner js-file-line">        scan_area <span class="pl-k">=</span> np.zeros(image_cones.shape, <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">bool</span>)</td>
      </tr>
      <tr>
        <td id="L187" class="blob-num js-line-number" data-line-number="187"></td>
        <td id="LC187" class="blob-code blob-code-inner js-file-line">        scan_area <span class="pl-k">=</span> set_hexagon(scan_area, center_scan, <span class="pl-v">r1</span><span class="pl-k">=</span><span class="pl-c1">self</span>.r1, <span class="pl-v">r2</span><span class="pl-k">=</span><span class="pl-c1">self</span>.r2, <span class="pl-v">value</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L188" class="blob-num js-line-number" data-line-number="188"></td>
        <td id="LC188" class="blob-code blob-code-inner js-file-line">        scan_result <span class="pl-k">=</span> np.zeros(image_cones.shape, <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">float</span>)</td>
      </tr>
      <tr>
        <td id="L189" class="blob-num js-line-number" data-line-number="189"></td>
        <td id="LC189" class="blob-code blob-code-inner js-file-line">        all_pixels_y, all_pixels_x <span class="pl-k">=</span> np.indices(scan_area.shape)</td>
      </tr>
      <tr>
        <td id="L190" class="blob-num js-line-number" data-line-number="190"></td>
        <td id="LC190" class="blob-code blob-code-inner js-file-line">        pixels_x <span class="pl-k">=</span> all_pixels_x[scan_area <span class="pl-k">==</span> <span class="pl-c1">1</span>].flatten()</td>
      </tr>
      <tr>
        <td id="L191" class="blob-num js-line-number" data-line-number="191"></td>
        <td id="LC191" class="blob-code blob-code-inner js-file-line">        pixels_y <span class="pl-k">=</span> all_pixels_y[scan_area <span class="pl-k">==</span> <span class="pl-c1">1</span>].flatten()</td>
      </tr>
      <tr>
        <td id="L192" class="blob-num js-line-number" data-line-number="192"></td>
        <td id="LC192" class="blob-code blob-code-inner js-file-line">        npixel <span class="pl-k">=</span> pixels_x.shape[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L193" class="blob-num js-line-number" data-line-number="193"></td>
        <td id="LC193" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>calculating hexagonalicity for each position in the pixel:<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L194" class="blob-num js-line-number" data-line-number="194"></td>
        <td id="LC194" class="blob-code blob-code-inner js-file-line">        last_precent <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L195" class="blob-num js-line-number" data-line-number="195"></td>
        <td id="LC195" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> pixel_x, pixel_y, <span class="pl-c1">iter</span> <span class="pl-k">in</span> <span class="pl-c1">zip</span>(pixels_x[::<span class="pl-c1">1</span>], pixels_y[::<span class="pl-c1">1</span>], <span class="pl-c1">range</span>(npixel)):</td>
      </tr>
      <tr>
        <td id="L196" class="blob-num js-line-number" data-line-number="196"></td>
        <td id="LC196" class="blob-code blob-code-inner js-file-line">            percent_done <span class="pl-k">=</span> np.floor((<span class="pl-c1">iter</span><span class="pl-k">+</span><span class="pl-c1">1</span>)<span class="pl-k">*</span><span class="pl-c1">100</span><span class="pl-k">/</span>npixel)</td>
      </tr>
      <tr>
        <td id="L197" class="blob-num js-line-number" data-line-number="197"></td>
        <td id="LC197" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> percent_done <span class="pl-k">&gt;</span> last_precent:</td>
      </tr>
      <tr>
        <td id="L198" class="blob-num js-line-number" data-line-number="198"></td>
        <td id="LC198" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">print</span>(percent_done, <span class="pl-s"><span class="pl-pds">&#39;</span>%<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L199" class="blob-num js-line-number" data-line-number="199"></td>
        <td id="LC199" class="blob-code blob-code-inner js-file-line">                last_precent <span class="pl-k">=</span> percent_done</td>
      </tr>
      <tr>
        <td id="L200" class="blob-num js-line-number" data-line-number="200"></td>
        <td id="LC200" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">hex</span> <span class="pl-k">=</span> <span class="pl-k">-</span>get_neg_hexagonalicity_with_mask((pixel_x, pixel_y), image_cones, <span class="pl-c1">self</span>.r1, <span class="pl-c1">self</span>.r2,</td>
      </tr>
      <tr>
        <td id="L201" class="blob-num js-line-number" data-line-number="201"></td>
        <td id="LC201" class="blob-code blob-code-inner js-file-line">                                                    <span class="pl-v">rotations</span><span class="pl-k">=</span>rotations)</td>
      </tr>
      <tr>
        <td id="L202" class="blob-num js-line-number" data-line-number="202"></td>
        <td id="LC202" class="blob-code blob-code-inner js-file-line">            scan_result[pixel_y, pixel_x] <span class="pl-k">=</span> <span class="pl-c1">hex</span></td>
      </tr>
      <tr>
        <td id="L203" class="blob-num js-line-number" data-line-number="203"></td>
        <td id="LC203" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L204" class="blob-num js-line-number" data-line-number="204"></td>
        <td id="LC204" class="blob-code blob-code-inner js-file-line">            plt.ioff()</td>
      </tr>
      <tr>
        <td id="L205" class="blob-num js-line-number" data-line-number="205"></td>
        <td id="LC205" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L206" class="blob-num js-line-number" data-line-number="206"></td>
        <td id="LC206" class="blob-code blob-code-inner js-file-line">            plt.ion()</td>
      </tr>
      <tr>
        <td id="L207" class="blob-num js-line-number" data-line-number="207"></td>
        <td id="LC207" class="blob-code blob-code-inner js-file-line">        fig <span class="pl-k">=</span> plt.figure()</td>
      </tr>
      <tr>
        <td id="L208" class="blob-num js-line-number" data-line-number="208"></td>
        <td id="LC208" class="blob-code blob-code-inner js-file-line">        ax <span class="pl-k">=</span> plt.subplot(<span class="pl-c1">1</span>, <span class="pl-c1">2</span>, <span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L209" class="blob-num js-line-number" data-line-number="209"></td>
        <td id="LC209" class="blob-code blob-code-inner js-file-line">        vmin <span class="pl-k">=</span> np.min(scan_result[scan_result<span class="pl-k">&gt;</span><span class="pl-c1">0</span>])</td>
      </tr>
      <tr>
        <td id="L210" class="blob-num js-line-number" data-line-number="210"></td>
        <td id="LC210" class="blob-code blob-code-inner js-file-line">        vmax<span class="pl-k">=</span> np.max(scan_result[scan_result<span class="pl-k">&gt;</span><span class="pl-c1">0</span>])</td>
      </tr>
      <tr>
        <td id="L211" class="blob-num js-line-number" data-line-number="211"></td>
        <td id="LC211" class="blob-code blob-code-inner js-file-line">        plt.imshow(scan_result, <span class="pl-v">cmap</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>gray<span class="pl-pds">&#39;</span></span>, <span class="pl-v">vmin</span><span class="pl-k">=</span>vmin, <span class="pl-v">vmax</span><span class="pl-k">=</span>vmax)</td>
      </tr>
      <tr>
        <td id="L212" class="blob-num js-line-number" data-line-number="212"></td>
        <td id="LC212" class="blob-code blob-code-inner js-file-line">        max_y, max_x <span class="pl-k">=</span> np.unravel_index(np.argmax(scan_result), <span class="pl-v">dims</span><span class="pl-k">=</span>scan_result.shape)</td>
      </tr>
      <tr>
        <td id="L213" class="blob-num js-line-number" data-line-number="213"></td>
        <td id="LC213" class="blob-code blob-code-inner js-file-line">        plt.plot(max_x, max_y, <span class="pl-s"><span class="pl-pds">&#39;</span>r+<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L214" class="blob-num js-line-number" data-line-number="214"></td>
        <td id="LC214" class="blob-code blob-code-inner js-file-line">        plt.xlim((np.min(pixels_x), np.max(pixels_x)))</td>
      </tr>
      <tr>
        <td id="L215" class="blob-num js-line-number" data-line-number="215"></td>
        <td id="LC215" class="blob-code blob-code-inner js-file-line">        plt.ylim((np.min(pixels_y), np.max(pixels_y)))</td>
      </tr>
      <tr>
        <td id="L216" class="blob-num js-line-number" data-line-number="216"></td>
        <td id="LC216" class="blob-code blob-code-inner js-file-line">        plt.grid(<span class="pl-c1">None</span>)</td>
      </tr>
      <tr>
        <td id="L217" class="blob-num js-line-number" data-line-number="217"></td>
        <td id="LC217" class="blob-code blob-code-inner js-file-line">        plt.axis(<span class="pl-s"><span class="pl-pds">&#39;</span>off<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L218" class="blob-num js-line-number" data-line-number="218"></td>
        <td id="LC218" class="blob-code blob-code-inner js-file-line">        ax.get_xaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L219" class="blob-num js-line-number" data-line-number="219"></td>
        <td id="LC219" class="blob-code blob-code-inner js-file-line">        ax.get_yaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L220" class="blob-num js-line-number" data-line-number="220"></td>
        <td id="LC220" class="blob-code blob-code-inner js-file-line">        ax <span class="pl-k">=</span> plt.subplot(<span class="pl-c1">1</span>, <span class="pl-c1">2</span>, <span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L221" class="blob-num js-line-number" data-line-number="221"></td>
        <td id="LC221" class="blob-code blob-code-inner js-file-line">        plt.imshow(image_cones <span class="pl-k">*</span> scan_area, <span class="pl-v">cmap</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>gray<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L222" class="blob-num js-line-number" data-line-number="222"></td>
        <td id="LC222" class="blob-code blob-code-inner js-file-line">        plt.plot(max_x, max_y, <span class="pl-s"><span class="pl-pds">&#39;</span>r+<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L223" class="blob-num js-line-number" data-line-number="223"></td>
        <td id="LC223" class="blob-code blob-code-inner js-file-line">        plt.xlim((np.min(pixels_x), np.max(pixels_x)))</td>
      </tr>
      <tr>
        <td id="L224" class="blob-num js-line-number" data-line-number="224"></td>
        <td id="LC224" class="blob-code blob-code-inner js-file-line">        plt.ylim((np.min(pixels_y), np.max(pixels_y)))</td>
      </tr>
      <tr>
        <td id="L225" class="blob-num js-line-number" data-line-number="225"></td>
        <td id="LC225" class="blob-code blob-code-inner js-file-line">        plt.grid(<span class="pl-c1">None</span>)</td>
      </tr>
      <tr>
        <td id="L226" class="blob-num js-line-number" data-line-number="226"></td>
        <td id="LC226" class="blob-code blob-code-inner js-file-line">        plt.axis(<span class="pl-s"><span class="pl-pds">&#39;</span>off<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L227" class="blob-num js-line-number" data-line-number="227"></td>
        <td id="LC227" class="blob-code blob-code-inner js-file-line">        ax.get_xaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L228" class="blob-num js-line-number" data-line-number="228"></td>
        <td id="LC228" class="blob-code blob-code-inner js-file-line">        ax.get_yaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L229" class="blob-num js-line-number" data-line-number="229"></td>
        <td id="LC229" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L230" class="blob-num js-line-number" data-line-number="230"></td>
        <td id="LC230" class="blob-code blob-code-inner js-file-line">            plt.show()</td>
      </tr>
      <tr>
        <td id="L231" class="blob-num js-line-number" data-line-number="231"></td>
        <td id="LC231" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L232" class="blob-num js-line-number" data-line-number="232"></td>
        <td id="LC232" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> <span class="pl-c1">self</span>.filename <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L233" class="blob-num js-line-number" data-line-number="233"></td>
        <td id="LC233" class="blob-code blob-code-inner js-file-line">                output_filename <span class="pl-k">=</span> <span class="pl-c1">self</span>.filename.replace(<span class="pl-s"><span class="pl-pds">&#39;</span>.fits<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>-hexagonalicity.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L234" class="blob-num js-line-number" data-line-number="234"></td>
        <td id="LC234" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L235" class="blob-num js-line-number" data-line-number="235"></td>
        <td id="LC235" class="blob-code blob-code-inner js-file-line">                output_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>hexagonalicity.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L236" class="blob-num js-line-number" data-line-number="236"></td>
        <td id="LC236" class="blob-code blob-code-inner js-file-line">            plt.savefig(output_filename, <span class="pl-v">bbox_inches</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>tight<span class="pl-pds">&#39;</span></span>, <span class="pl-v">pad_inches</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L237" class="blob-num js-line-number" data-line-number="237"></td>
        <td id="LC237" class="blob-code blob-code-inner js-file-line">            plt.close(fig)</td>
      </tr>
      <tr>
        <td id="L238" class="blob-num js-line-number" data-line-number="238"></td>
        <td id="LC238" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(output_filename, <span class="pl-s"><span class="pl-pds">&#39;</span>saved.<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L239" class="blob-num js-line-number" data-line-number="239"></td>
        <td id="LC239" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L240" class="blob-num js-line-number" data-line-number="240"></td>
        <td id="LC240" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_cone</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">radius_mask</span>, <span class="pl-smi">output_dir</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">save_to_file</span><span class="pl-k">=</span><span class="pl-c1">True</span>):</td>
      </tr>
      <tr>
        <td id="L241" class="blob-num js-line-number" data-line-number="241"></td>
        <td id="LC241" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L242" class="blob-num js-line-number" data-line-number="242"></td>
        <td id="LC242" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        function to find the center of a camera&#39;s pixel</span></td>
      </tr>
      <tr>
        <td id="L243" class="blob-num js-line-number" data-line-number="243"></td>
        <td id="LC243" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param radius_mask: radius of the mask used for extracting pixels from image (see plot_cones())</span></td>
      </tr>
      <tr>
        <td id="L244" class="blob-num js-line-number" data-line-number="244"></td>
        <td id="LC244" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param output_dir: optional directory where to put the resulting image. If None (default) the image is displayed</span></td>
      </tr>
      <tr>
        <td id="L245" class="blob-num js-line-number" data-line-number="245"></td>
        <td id="LC245" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        instead of being saved to a file.</span></td>
      </tr>
      <tr>
        <td id="L246" class="blob-num js-line-number" data-line-number="246"></td>
        <td id="LC246" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param save_to_file: boolean (default = True), should the resulting image be saved to a fit files along</span></td>
      </tr>
      <tr>
        <td id="L247" class="blob-num js-line-number" data-line-number="247"></td>
        <td id="LC247" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        with important parameters ?</span></td>
      </tr>
      <tr>
        <td id="L248" class="blob-num js-line-number" data-line-number="248"></td>
        <td id="LC248" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L249" class="blob-num js-line-number" data-line-number="249"></td>
        <td id="LC249" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">type</span>(radius_mask) <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">float</span>:</td>
      </tr>
      <tr>
        <td id="L250" class="blob-num js-line-number" data-line-number="250"></td>
        <td id="LC250" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">raise</span> <span class="pl-c1">AttributeError</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>radius_mask must be a float.<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L251" class="blob-num js-line-number" data-line-number="251"></td>
        <td id="LC251" class="blob-code blob-code-inner js-file-line">        mask_fft <span class="pl-k">=</span> <span class="pl-c1">self</span>.get_fft_mask(<span class="pl-v">radius</span><span class="pl-k">=</span>radius_mask)</td>
      </tr>
      <tr>
        <td id="L252" class="blob-num js-line-number" data-line-number="252"></td>
        <td id="LC252" class="blob-code blob-code-inner js-file-line">        image_cones <span class="pl-k">=</span> np.real(np.fft.ifft2(<span class="pl-c1">self</span>.fft_image_cones <span class="pl-k">*</span> mask_fft))</td>
      </tr>
      <tr>
        <td id="L253" class="blob-num js-line-number" data-line-number="253"></td>
        <td id="LC253" class="blob-code blob-code-inner js-file-line">        center_image <span class="pl-k">=</span> (np.array(image_cones.shape[::<span class="pl-k">-</span><span class="pl-c1">1</span>]) <span class="pl-k">-</span> <span class="pl-c1">1</span>) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L254" class="blob-num js-line-number" data-line-number="254"></td>
        <td id="LC254" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">self</span>.center_fitted <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L255" class="blob-num js-line-number" data-line-number="255"></td>
        <td id="LC255" class="blob-code blob-code-inner js-file-line">            center_tested <span class="pl-k">=</span> center_image</td>
      </tr>
      <tr>
        <td id="L256" class="blob-num js-line-number" data-line-number="256"></td>
        <td id="LC256" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L257" class="blob-num js-line-number" data-line-number="257"></td>
        <td id="LC257" class="blob-code blob-code-inner js-file-line">            center_tested <span class="pl-k">=</span> <span class="pl-c1">self</span>.center_fitted</td>
      </tr>
      <tr>
        <td id="L258" class="blob-num js-line-number" data-line-number="258"></td>
        <td id="LC258" class="blob-code blob-code-inner js-file-line">        points <span class="pl-k">=</span> np.array((center_tested <span class="pl-k">+</span> <span class="pl-c1">self</span>.r1,</td>
      </tr>
      <tr>
        <td id="L259" class="blob-num js-line-number" data-line-number="259"></td>
        <td id="LC259" class="blob-code blob-code-inner js-file-line">                           center_tested <span class="pl-k">+</span> <span class="pl-c1">self</span>.r2,</td>
      </tr>
      <tr>
        <td id="L260" class="blob-num js-line-number" data-line-number="260"></td>
        <td id="LC260" class="blob-code blob-code-inner js-file-line">                           center_tested <span class="pl-k">+</span> <span class="pl-c1">self</span>.r3,</td>
      </tr>
      <tr>
        <td id="L261" class="blob-num js-line-number" data-line-number="261"></td>
        <td id="LC261" class="blob-code blob-code-inner js-file-line">                           center_tested <span class="pl-k">-</span> <span class="pl-c1">self</span>.r1,</td>
      </tr>
      <tr>
        <td id="L262" class="blob-num js-line-number" data-line-number="262"></td>
        <td id="LC262" class="blob-code blob-code-inner js-file-line">                           center_tested <span class="pl-k">-</span> <span class="pl-c1">self</span>.r2,</td>
      </tr>
      <tr>
        <td id="L263" class="blob-num js-line-number" data-line-number="263"></td>
        <td id="LC263" class="blob-code blob-code-inner js-file-line">                           center_tested <span class="pl-k">-</span> <span class="pl-c1">self</span>.r3))</td>
      </tr>
      <tr>
        <td id="L264" class="blob-num js-line-number" data-line-number="264"></td>
        <td id="LC264" class="blob-code blob-code-inner js-file-line">        bounds <span class="pl-k">=</span> ((np.min(points[:, <span class="pl-c1">0</span>]), np.max(points[:, <span class="pl-c1">0</span>])),</td>
      </tr>
      <tr>
        <td id="L265" class="blob-num js-line-number" data-line-number="265"></td>
        <td id="LC265" class="blob-code blob-code-inner js-file-line">                  (np.min(points[:, <span class="pl-c1">1</span>]), np.max(points[:, <span class="pl-c1">1</span>])))</td>
      </tr>
      <tr>
        <td id="L266" class="blob-num js-line-number" data-line-number="266"></td>
        <td id="LC266" class="blob-code blob-code-inner js-file-line">        offsets<span class="pl-k">=</span> (<span class="pl-c1">0</span><span class="pl-k">*</span><span class="pl-c1">self</span>.r1, (<span class="pl-c1">self</span>.r1<span class="pl-k">+</span><span class="pl-c1">self</span>.r2)<span class="pl-k">/</span><span class="pl-c1">3</span>, <span class="pl-c1">self</span>.r1, <span class="pl-c1">self</span>.r2, (<span class="pl-c1">self</span>.r2<span class="pl-k">+</span><span class="pl-c1">self</span>.r3)<span class="pl-k">/</span><span class="pl-c1">3</span>, <span class="pl-c1">self</span>.r3,</td>
      </tr>
      <tr>
        <td id="L267" class="blob-num js-line-number" data-line-number="267"></td>
        <td id="LC267" class="blob-code blob-code-inner js-file-line">                  (<span class="pl-c1">self</span>.r3<span class="pl-k">-</span><span class="pl-c1">self</span>.r1)<span class="pl-k">/</span><span class="pl-c1">3</span>, <span class="pl-k">-</span><span class="pl-c1">self</span>.r1, (<span class="pl-k">-</span><span class="pl-c1">self</span>.r1<span class="pl-k">-</span><span class="pl-c1">self</span>.r2)<span class="pl-k">/</span><span class="pl-c1">3</span>, <span class="pl-k">-</span><span class="pl-c1">self</span>.r2,</td>
      </tr>
      <tr>
        <td id="L268" class="blob-num js-line-number" data-line-number="268"></td>
        <td id="LC268" class="blob-code blob-code-inner js-file-line">                  (<span class="pl-k">-</span><span class="pl-c1">self</span>.r2<span class="pl-k">-</span><span class="pl-c1">self</span>.r3)<span class="pl-k">/</span><span class="pl-c1">3</span>, <span class="pl-k">-</span><span class="pl-c1">self</span>.r3, (<span class="pl-k">-</span><span class="pl-c1">self</span>.r3<span class="pl-k">+</span><span class="pl-c1">self</span>.r1)<span class="pl-k">/</span><span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L269" class="blob-num js-line-number" data-line-number="269"></td>
        <td id="LC269" class="blob-code blob-code-inner js-file-line">        results<span class="pl-k">=</span>[]</td>
      </tr>
      <tr>
        <td id="L270" class="blob-num js-line-number" data-line-number="270"></td>
        <td id="LC270" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> i,offset <span class="pl-k">in</span> <span class="pl-c1">enumerate</span>(offsets):</td>
      </tr>
      <tr>
        <td id="L271" class="blob-num js-line-number" data-line-number="271"></td>
        <td id="LC271" class="blob-code blob-code-inner js-file-line">            res <span class="pl-k">=</span> optimize.minimize(get_neg_hexagonalicity_with_mask,</td>
      </tr>
      <tr>
        <td id="L272" class="blob-num js-line-number" data-line-number="272"></td>
        <td id="LC272" class="blob-code blob-code-inner js-file-line">                                    center_tested<span class="pl-k">+</span>offset, <span class="pl-v">args</span><span class="pl-k">=</span>(image_cones, <span class="pl-c1">self</span>.r1, <span class="pl-c1">self</span>.r2, (<span class="pl-c1">60</span>, <span class="pl-c1">300</span>)),</td>
      </tr>
      <tr>
        <td id="L273" class="blob-num js-line-number" data-line-number="273"></td>
        <td id="LC273" class="blob-code blob-code-inner js-file-line">                                    <span class="pl-v">bounds</span><span class="pl-k">=</span>bounds, <span class="pl-v">method</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>TNC<span class="pl-pds">&#39;</span></span>,</td>
      </tr>
      <tr>
        <td id="L274" class="blob-num js-line-number" data-line-number="274"></td>
        <td id="LC274" class="blob-code blob-code-inner js-file-line">                                    <span class="pl-v">options</span><span class="pl-k">=</span>{<span class="pl-s"><span class="pl-pds">&#39;</span>disp<span class="pl-pds">&#39;</span></span>: <span class="pl-c1">False</span>, <span class="pl-s"><span class="pl-pds">&#39;</span>eps<span class="pl-pds">&#39;</span></span>: <span class="pl-c1">1</span>, <span class="pl-s"><span class="pl-pds">&#39;</span>xtol<span class="pl-pds">&#39;</span></span>: <span class="pl-c1">1e-2</span>, <span class="pl-s"><span class="pl-pds">&#39;</span>maxiter<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">200</span>})</td>
      </tr>
      <tr>
        <td id="L275" class="blob-num js-line-number" data-line-number="275"></td>
        <td id="LC275" class="blob-code blob-code-inner js-file-line">            results.append(res)</td>
      </tr>
      <tr>
        <td id="L276" class="blob-num js-line-number" data-line-number="276"></td>
        <td id="LC276" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>fit<span class="pl-pds">&#39;</span></span>, i<span class="pl-k">+</span><span class="pl-c1">1</span>, <span class="pl-s"><span class="pl-pds">&#39;</span>/<span class="pl-pds">&#39;</span></span>, <span class="pl-c1">len</span>(offsets))</td>
      </tr>
      <tr>
        <td id="L277" class="blob-num js-line-number" data-line-number="277"></td>
        <td id="LC277" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> <span class="pl-k">-</span>res.fun <span class="pl-k">&gt;</span> <span class="pl-c1">0.5</span>:</td>
      </tr>
      <tr>
        <td id="L278" class="blob-num js-line-number" data-line-number="278"></td>
        <td id="LC278" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">break</span></td>
      </tr>
      <tr>
        <td id="L279" class="blob-num js-line-number" data-line-number="279"></td>
        <td id="LC279" class="blob-code blob-code-inner js-file-line">        hex_results <span class="pl-k">=</span> np.array([<span class="pl-k">-</span>res.fun <span class="pl-k">for</span> res <span class="pl-k">in</span> results])</td>
      </tr>
      <tr>
        <td id="L280" class="blob-num js-line-number" data-line-number="280"></td>
        <td id="LC280" class="blob-code blob-code-inner js-file-line">        pos_results <span class="pl-k">=</span> np.array([res.x <span class="pl-k">for</span> res <span class="pl-k">in</span> results])</td>
      </tr>
      <tr>
        <td id="L281" class="blob-num js-line-number" data-line-number="281"></td>
        <td id="LC281" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.center_fitted <span class="pl-k">=</span> pos_results[np.argmax(hex_results)]</td>
      </tr>
      <tr>
        <td id="L282" class="blob-num js-line-number" data-line-number="282"></td>
        <td id="LC282" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span>fine fit for best result</span></td>
      </tr>
      <tr>
        <td id="L283" class="blob-num js-line-number" data-line-number="283"></td>
        <td id="LC283" class="blob-code blob-code-inner js-file-line">        res <span class="pl-k">=</span> optimize.minimize(get_neg_hexagonalicity_with_mask,</td>
      </tr>
      <tr>
        <td id="L284" class="blob-num js-line-number" data-line-number="284"></td>
        <td id="LC284" class="blob-code blob-code-inner js-file-line">                                <span class="pl-c1">self</span>.center_fitted, <span class="pl-v">args</span><span class="pl-k">=</span>(image_cones, <span class="pl-c1">self</span>.r1, <span class="pl-c1">self</span>.r2, (<span class="pl-c1">60</span>, <span class="pl-c1">120</span>, <span class="pl-c1">180</span>, <span class="pl-c1">240</span>, <span class="pl-c1">300</span>)),</td>
      </tr>
      <tr>
        <td id="L285" class="blob-num js-line-number" data-line-number="285"></td>
        <td id="LC285" class="blob-code blob-code-inner js-file-line">                                <span class="pl-v">bounds</span><span class="pl-k">=</span>bounds, <span class="pl-v">method</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>TNC<span class="pl-pds">&#39;</span></span>,</td>
      </tr>
      <tr>
        <td id="L286" class="blob-num js-line-number" data-line-number="286"></td>
        <td id="LC286" class="blob-code blob-code-inner js-file-line">                                <span class="pl-v">options</span><span class="pl-k">=</span>{<span class="pl-s"><span class="pl-pds">&#39;</span>disp<span class="pl-pds">&#39;</span></span>: <span class="pl-c1">False</span>, <span class="pl-s"><span class="pl-pds">&#39;</span>eps<span class="pl-pds">&#39;</span></span>: <span class="pl-c1">.01</span>, <span class="pl-s"><span class="pl-pds">&#39;</span>xtol<span class="pl-pds">&#39;</span></span>: <span class="pl-c1">1e-4</span>, <span class="pl-s"><span class="pl-pds">&#39;</span>maxiter<span class="pl-pds">&#39;</span></span>: <span class="pl-c1">200</span>})</td>
      </tr>
      <tr>
        <td id="L287" class="blob-num js-line-number" data-line-number="287"></td>
        <td id="LC287" class="blob-code blob-code-inner js-file-line">        results.append(res)</td>
      </tr>
      <tr>
        <td id="L288" class="blob-num js-line-number" data-line-number="288"></td>
        <td id="LC288" class="blob-code blob-code-inner js-file-line">        hex_results <span class="pl-k">=</span> np.array([<span class="pl-k">-</span>res.fun <span class="pl-k">for</span> res <span class="pl-k">in</span> results])</td>
      </tr>
      <tr>
        <td id="L289" class="blob-num js-line-number" data-line-number="289"></td>
        <td id="LC289" class="blob-code blob-code-inner js-file-line">        pos_results <span class="pl-k">=</span> np.array([res.x <span class="pl-k">for</span> res <span class="pl-k">in</span> results])</td>
      </tr>
      <tr>
        <td id="L290" class="blob-num js-line-number" data-line-number="290"></td>
        <td id="LC290" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.center_fitted <span class="pl-k">=</span> pos_results[np.argmax(hex_results)]</td>
      </tr>
      <tr>
        <td id="L291" class="blob-num js-line-number" data-line-number="291"></td>
        <td id="LC291" class="blob-code blob-code-inner js-file-line">        hex_result <span class="pl-k">=</span> hex_results[np.argmax(hex_results)]</td>
      </tr>
      <tr>
        <td id="L292" class="blob-num js-line-number" data-line-number="292"></td>
        <td id="LC292" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>pixel center found: <span class="pl-pds">&#39;</span></span>, <span class="pl-c1">self</span>.center_fitted, <span class="pl-s"><span class="pl-pds">&#39;</span>with hex=<span class="pl-pds">&#39;</span></span>, hex_result)</td>
      </tr>
      <tr>
        <td id="L293" class="blob-num js-line-number" data-line-number="293"></td>
        <td id="LC293" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-c1">self</span>.center_fitted <span class="pl-k">-</span> center_image, <span class="pl-s"><span class="pl-pds">&#39;</span>px from center<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L294" class="blob-num js-line-number" data-line-number="294"></td>
        <td id="LC294" class="blob-code blob-code-inner js-file-line">        points_fitted <span class="pl-k">=</span> points <span class="pl-k">-</span> center_tested <span class="pl-k">+</span> np.array(<span class="pl-c1">self</span>.center_fitted)</td>
      </tr>
      <tr>
        <td id="L295" class="blob-num js-line-number" data-line-number="295"></td>
        <td id="LC295" class="blob-code blob-code-inner js-file-line">        pixels_x_min <span class="pl-k">=</span> <span class="pl-c1">int</span>(np.floor(np.min(points_fitted[:, <span class="pl-c1">0</span>])))</td>
      </tr>
      <tr>
        <td id="L296" class="blob-num js-line-number" data-line-number="296"></td>
        <td id="LC296" class="blob-code blob-code-inner js-file-line">        pixels_x_max <span class="pl-k">=</span> <span class="pl-c1">int</span>(np.ceil(np.max(points_fitted[:, <span class="pl-c1">0</span>]))) <span class="pl-k">+</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L297" class="blob-num js-line-number" data-line-number="297"></td>
        <td id="LC297" class="blob-code blob-code-inner js-file-line">        pixels_y_min <span class="pl-k">=</span> <span class="pl-c1">int</span>(np.floor(np.min(points_fitted[:, <span class="pl-c1">1</span>])))</td>
      </tr>
      <tr>
        <td id="L298" class="blob-num js-line-number" data-line-number="298"></td>
        <td id="LC298" class="blob-code blob-code-inner js-file-line">        pixels_y_max <span class="pl-k">=</span> <span class="pl-c1">int</span>(np.ceil(np.max(points_fitted[:, <span class="pl-c1">1</span>]))) <span class="pl-k">+</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L299" class="blob-num js-line-number" data-line-number="299"></td>
        <td id="LC299" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span>plot_image(image_cones)</span></td>
      </tr>
      <tr>
        <td id="L300" class="blob-num js-line-number" data-line-number="300"></td>
        <td id="LC300" class="blob-code blob-code-inner js-file-line">        image_crop <span class="pl-k">=</span> image_cones[pixels_y_min:pixels_y_max, pixels_x_min:pixels_x_max]</td>
      </tr>
      <tr>
        <td id="L301" class="blob-num js-line-number" data-line-number="301"></td>
        <td id="LC301" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span>plot_image(image_crop)</span></td>
      </tr>
      <tr>
        <td id="L302" class="blob-num js-line-number" data-line-number="302"></td>
        <td id="LC302" class="blob-code blob-code-inner js-file-line">        center_crop <span class="pl-k">=</span> np.array(<span class="pl-c1">self</span>.center_fitted) <span class="pl-k">-</span> np.array((pixels_x_min, pixels_y_min))</td>
      </tr>
      <tr>
        <td id="L303" class="blob-num js-line-number" data-line-number="303"></td>
        <td id="LC303" class="blob-code blob-code-inner js-file-line">        mask_hexa <span class="pl-k">=</span> np.zeros_like(image_crop)</td>
      </tr>
      <tr>
        <td id="L304" class="blob-num js-line-number" data-line-number="304"></td>
        <td id="LC304" class="blob-code blob-code-inner js-file-line">        mask_hexa <span class="pl-k">=</span> set_hexagon(mask_hexa, <span class="pl-v">center</span><span class="pl-k">=</span>center_crop, <span class="pl-v">r1</span><span class="pl-k">=</span><span class="pl-c1">self</span>.r1, <span class="pl-v">r2</span><span class="pl-k">=</span><span class="pl-c1">self</span>.r2)</td>
      </tr>
      <tr>
        <td id="L305" class="blob-num js-line-number" data-line-number="305"></td>
        <td id="LC305" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.image_cone <span class="pl-k">=</span> image_crop <span class="pl-k">*</span> mask_hexa</td>
      </tr>
      <tr>
        <td id="L306" class="blob-num js-line-number" data-line-number="306"></td>
        <td id="LC306" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> save_to_file:</td>
      </tr>
      <tr>
        <td id="L307" class="blob-num js-line-number" data-line-number="307"></td>
        <td id="LC307" class="blob-code blob-code-inner js-file-line">            hdu <span class="pl-k">=</span> fits.PrimaryHDU(<span class="pl-c1">self</span>.image_cone)</td>
      </tr>
      <tr>
        <td id="L308" class="blob-num js-line-number" data-line-number="308"></td>
        <td id="LC308" class="blob-code blob-code-inner js-file-line">            hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>center<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.center_fitted[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.center_fitted[<span class="pl-c1">1</span>],</td>
      </tr>
      <tr>
        <td id="L309" class="blob-num js-line-number" data-line-number="309"></td>
        <td id="LC309" class="blob-code blob-code-inner js-file-line">                                      <span class="pl-s"><span class="pl-pds">&#39;</span>fitted position (in original image) of the hexagon center<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L310" class="blob-num js-line-number" data-line-number="310"></td>
        <td id="LC310" class="blob-code blob-code-inner js-file-line">            hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>ks1<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.ks[<span class="pl-c1">0</span>, <span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.ks[<span class="pl-c1">0</span>, <span class="pl-c1">1</span>],</td>
      </tr>
      <tr>
        <td id="L311" class="blob-num js-line-number" data-line-number="311"></td>
        <td id="LC311" class="blob-code blob-code-inner js-file-line">                                 <span class="pl-s"><span class="pl-pds">&#39;</span>1st vector of the base of the hexagonal lattice in reciprocal space<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L312" class="blob-num js-line-number" data-line-number="312"></td>
        <td id="LC312" class="blob-code blob-code-inner js-file-line">            hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>ks2<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.ks[<span class="pl-c1">1</span> ,<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.ks[<span class="pl-c1">1</span>, <span class="pl-c1">1</span>],</td>
      </tr>
      <tr>
        <td id="L313" class="blob-num js-line-number" data-line-number="313"></td>
        <td id="LC313" class="blob-code blob-code-inner js-file-line">                                 <span class="pl-s"><span class="pl-pds">&#39;</span>2nd vector of the base of the hexagonal lattice in reciprocal space<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L314" class="blob-num js-line-number" data-line-number="314"></td>
        <td id="LC314" class="blob-code blob-code-inner js-file-line">            hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>v1<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.v1_lattice[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.v1_lattice[<span class="pl-c1">1</span>],</td>
      </tr>
      <tr>
        <td id="L315" class="blob-num js-line-number" data-line-number="315"></td>
        <td id="LC315" class="blob-code blob-code-inner js-file-line">                                <span class="pl-s"><span class="pl-pds">&#39;</span>spacing between 2 hexagons along the 1st axis<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L316" class="blob-num js-line-number" data-line-number="316"></td>
        <td id="LC316" class="blob-code blob-code-inner js-file-line">            hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>v2<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.v2_lattice[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.v2_lattice[<span class="pl-c1">1</span>],</td>
      </tr>
      <tr>
        <td id="L317" class="blob-num js-line-number" data-line-number="317"></td>
        <td id="LC317" class="blob-code blob-code-inner js-file-line">                                <span class="pl-s"><span class="pl-pds">&#39;</span>spacing between 2 hexagons along the 2nd axis<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L318" class="blob-num js-line-number" data-line-number="318"></td>
        <td id="LC318" class="blob-code blob-code-inner js-file-line">            hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>r1<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.r1[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.r1[<span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>1st radius of the hexagon<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L319" class="blob-num js-line-number" data-line-number="319"></td>
        <td id="LC319" class="blob-code blob-code-inner js-file-line">            hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>r2<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.r2[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.r2[<span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>2nd radius of the hexagon<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L320" class="blob-num js-line-number" data-line-number="320"></td>
        <td id="LC320" class="blob-code blob-code-inner js-file-line">            hdu.header[<span class="pl-s"><span class="pl-pds">&#39;</span>r3<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.r3[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.r3[<span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>3rd radius of the hexagon<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L321" class="blob-num js-line-number" data-line-number="321"></td>
        <td id="LC321" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> <span class="pl-c1">self</span>.filename <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L322" class="blob-num js-line-number" data-line-number="322"></td>
        <td id="LC322" class="blob-code blob-code-inner js-file-line">                cone_filename <span class="pl-k">=</span> <span class="pl-c1">self</span>.filename.replace(<span class="pl-s"><span class="pl-pds">&#39;</span>.fits<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>-cone.fits<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L323" class="blob-num js-line-number" data-line-number="323"></td>
        <td id="LC323" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L324" class="blob-num js-line-number" data-line-number="324"></td>
        <td id="LC324" class="blob-code blob-code-inner js-file-line">                cone_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>cone.fits<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L325" class="blob-num js-line-number" data-line-number="325"></td>
        <td id="LC325" class="blob-code blob-code-inner js-file-line">            hdu.writeto(cone_filename, <span class="pl-v">overwrite</span><span class="pl-k">=</span><span class="pl-c1">True</span>)</td>
      </tr>
      <tr>
        <td id="L326" class="blob-num js-line-number" data-line-number="326"></td>
        <td id="LC326" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>cone saved to <span class="pl-pds">&#39;</span></span>, cone_filename)</td>
      </tr>
      <tr>
        <td id="L327" class="blob-num js-line-number" data-line-number="327"></td>
        <td id="LC327" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L328" class="blob-num js-line-number" data-line-number="328"></td>
        <td id="LC328" class="blob-code blob-code-inner js-file-line">            plt.ioff()</td>
      </tr>
      <tr>
        <td id="L329" class="blob-num js-line-number" data-line-number="329"></td>
        <td id="LC329" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L330" class="blob-num js-line-number" data-line-number="330"></td>
        <td id="LC330" class="blob-code blob-code-inner js-file-line">            plt.ion()</td>
      </tr>
      <tr>
        <td id="L331" class="blob-num js-line-number" data-line-number="331"></td>
        <td id="LC331" class="blob-code blob-code-inner js-file-line">        fig <span class="pl-k">=</span> plt.figure()</td>
      </tr>
      <tr>
        <td id="L332" class="blob-num js-line-number" data-line-number="332"></td>
        <td id="LC332" class="blob-code blob-code-inner js-file-line">        ax <span class="pl-k">=</span> plt.gca()</td>
      </tr>
      <tr>
        <td id="L333" class="blob-num js-line-number" data-line-number="333"></td>
        <td id="LC333" class="blob-code blob-code-inner js-file-line">        plt.imshow(<span class="pl-c1">self</span>.image_cone, <span class="pl-v">cmap</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>gray<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L334" class="blob-num js-line-number" data-line-number="334"></td>
        <td id="LC334" class="blob-code blob-code-inner js-file-line">        plt.plot(<span class="pl-c1">self</span>.center_fitted[<span class="pl-c1">0</span>]<span class="pl-k">-</span>pixels_x_min,<span class="pl-c1">self</span>.center_fitted[<span class="pl-c1">1</span>]<span class="pl-k">-</span>pixels_y_min,<span class="pl-s"><span class="pl-pds">&#39;</span>y+<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L335" class="blob-num js-line-number" data-line-number="335"></td>
        <td id="LC335" class="blob-code blob-code-inner js-file-line">        plt.xlim((<span class="pl-k">-</span><span class="pl-c1">0.5</span>, pixels_x_max<span class="pl-k">-</span>pixels_x_min<span class="pl-k">-</span><span class="pl-c1">0.5</span>))</td>
      </tr>
      <tr>
        <td id="L336" class="blob-num js-line-number" data-line-number="336"></td>
        <td id="LC336" class="blob-code blob-code-inner js-file-line">        plt.ylim((<span class="pl-k">-</span><span class="pl-c1">0.5</span>, pixels_y_max<span class="pl-k">-</span>pixels_y_min<span class="pl-k">-</span><span class="pl-c1">0.5</span>))</td>
      </tr>
      <tr>
        <td id="L337" class="blob-num js-line-number" data-line-number="337"></td>
        <td id="LC337" class="blob-code blob-code-inner js-file-line">        plt.grid(<span class="pl-c1">None</span>)</td>
      </tr>
      <tr>
        <td id="L338" class="blob-num js-line-number" data-line-number="338"></td>
        <td id="LC338" class="blob-code blob-code-inner js-file-line">        plt.axis(<span class="pl-s"><span class="pl-pds">&#39;</span>off<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L339" class="blob-num js-line-number" data-line-number="339"></td>
        <td id="LC339" class="blob-code blob-code-inner js-file-line">        ax.get_xaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L340" class="blob-num js-line-number" data-line-number="340"></td>
        <td id="LC340" class="blob-code blob-code-inner js-file-line">        ax.get_yaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L341" class="blob-num js-line-number" data-line-number="341"></td>
        <td id="LC341" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L342" class="blob-num js-line-number" data-line-number="342"></td>
        <td id="LC342" class="blob-code blob-code-inner js-file-line">            plt.show()</td>
      </tr>
      <tr>
        <td id="L343" class="blob-num js-line-number" data-line-number="343"></td>
        <td id="LC343" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L344" class="blob-num js-line-number" data-line-number="344"></td>
        <td id="LC344" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> <span class="pl-c1">self</span>.filename <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L345" class="blob-num js-line-number" data-line-number="345"></td>
        <td id="LC345" class="blob-code blob-code-inner js-file-line">                output_filename <span class="pl-k">=</span> <span class="pl-c1">self</span>.filename.replace(<span class="pl-s"><span class="pl-pds">&#39;</span>.fits<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>-cone.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L346" class="blob-num js-line-number" data-line-number="346"></td>
        <td id="LC346" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L347" class="blob-num js-line-number" data-line-number="347"></td>
        <td id="LC347" class="blob-code blob-code-inner js-file-line">                output_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>cone.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L348" class="blob-num js-line-number" data-line-number="348"></td>
        <td id="LC348" class="blob-code blob-code-inner js-file-line">            plt.savefig(output_filename, <span class="pl-v">bbox_inches</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>tight<span class="pl-pds">&#39;</span></span>, <span class="pl-v">pad_inches</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L349" class="blob-num js-line-number" data-line-number="349"></td>
        <td id="LC349" class="blob-code blob-code-inner js-file-line">            plt.close(fig)</td>
      </tr>
      <tr>
        <td id="L350" class="blob-num js-line-number" data-line-number="350"></td>
        <td id="LC350" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(output_filename, <span class="pl-s"><span class="pl-pds">&#39;</span>saved.<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L351" class="blob-num js-line-number" data-line-number="351"></td>
        <td id="LC351" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L352" class="blob-num js-line-number" data-line-number="352"></td>
        <td id="LC352" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">plot_fft_cones</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">radius_mask</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">output_dir</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L353" class="blob-num js-line-number" data-line-number="353"></td>
        <td id="LC353" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L354" class="blob-num js-line-number" data-line-number="354"></td>
        <td id="LC354" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        plot the FFT of the filtered lid CCD image. If radius_mask is given, everything more distant than it from</span></td>
      </tr>
      <tr>
        <td id="L355" class="blob-num js-line-number" data-line-number="355"></td>
        <td id="LC355" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        a peak is set to 0.</span></td>
      </tr>
      <tr>
        <td id="L356" class="blob-num js-line-number" data-line-number="356"></td>
        <td id="LC356" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param radius_mask: radius in pixels of circular mask around each peak. Default = no mask.</span></td>
      </tr>
      <tr>
        <td id="L357" class="blob-num js-line-number" data-line-number="357"></td>
        <td id="LC357" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param output_dir: optional directory where to put the resulting image. If None (default) the image is displayed</span></td>
      </tr>
      <tr>
        <td id="L358" class="blob-num js-line-number" data-line-number="358"></td>
        <td id="LC358" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        instead of being saved to a file.</span></td>
      </tr>
      <tr>
        <td id="L359" class="blob-num js-line-number" data-line-number="359"></td>
        <td id="LC359" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L360" class="blob-num js-line-number" data-line-number="360"></td>
        <td id="LC360" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L361" class="blob-num js-line-number" data-line-number="361"></td>
        <td id="LC361" class="blob-code blob-code-inner js-file-line">            plt.ioff()</td>
      </tr>
      <tr>
        <td id="L362" class="blob-num js-line-number" data-line-number="362"></td>
        <td id="LC362" class="blob-code blob-code-inner js-file-line">        fig <span class="pl-k">=</span> plt.figure()</td>
      </tr>
      <tr>
        <td id="L363" class="blob-num js-line-number" data-line-number="363"></td>
        <td id="LC363" class="blob-code blob-code-inner js-file-line">        ax <span class="pl-k">=</span> fig.add_subplot(<span class="pl-c1">111</span>)</td>
      </tr>
      <tr>
        <td id="L364" class="blob-num js-line-number" data-line-number="364"></td>
        <td id="LC364" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> radius_mask <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L365" class="blob-num js-line-number" data-line-number="365"></td>
        <td id="LC365" class="blob-code blob-code-inner js-file-line">            mask <span class="pl-k">=</span> <span class="pl-c1">self</span>.get_fft_mask(<span class="pl-v">radius</span><span class="pl-k">=</span>radius_mask)</td>
      </tr>
      <tr>
        <td id="L366" class="blob-num js-line-number" data-line-number="366"></td>
        <td id="LC366" class="blob-code blob-code-inner js-file-line">            fft_image_cones <span class="pl-k">=</span> <span class="pl-c1">self</span>.fft_image_cones <span class="pl-k">*</span> mask</td>
      </tr>
      <tr>
        <td id="L367" class="blob-num js-line-number" data-line-number="367"></td>
        <td id="LC367" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L368" class="blob-num js-line-number" data-line-number="368"></td>
        <td id="LC368" class="blob-code blob-code-inner js-file-line">            fft_image_cones <span class="pl-k">=</span> <span class="pl-c1">self</span>.fft_image_cones</td>
      </tr>
      <tr>
        <td id="L369" class="blob-num js-line-number" data-line-number="369"></td>
        <td id="LC369" class="blob-code blob-code-inner js-file-line">        plt.imshow(np.abs(np.fft.fftshift(fft_image_cones)), <span class="pl-v">cmap</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>gray<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L370" class="blob-num js-line-number" data-line-number="370"></td>
        <td id="LC370" class="blob-code blob-code-inner js-file-line">        plt.grid(<span class="pl-c1">None</span>)</td>
      </tr>
      <tr>
        <td id="L371" class="blob-num js-line-number" data-line-number="371"></td>
        <td id="LC371" class="blob-code blob-code-inner js-file-line">        plt.axis(<span class="pl-s"><span class="pl-pds">&#39;</span>off<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L372" class="blob-num js-line-number" data-line-number="372"></td>
        <td id="LC372" class="blob-code blob-code-inner js-file-line">        ax.get_xaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L373" class="blob-num js-line-number" data-line-number="373"></td>
        <td id="LC373" class="blob-code blob-code-inner js-file-line">        ax.get_yaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L374" class="blob-num js-line-number" data-line-number="374"></td>
        <td id="LC374" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L375" class="blob-num js-line-number" data-line-number="375"></td>
        <td id="LC375" class="blob-code blob-code-inner js-file-line">            plt.show()</td>
      </tr>
      <tr>
        <td id="L376" class="blob-num js-line-number" data-line-number="376"></td>
        <td id="LC376" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L377" class="blob-num js-line-number" data-line-number="377"></td>
        <td id="LC377" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> <span class="pl-c1">self</span>.filename <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L378" class="blob-num js-line-number" data-line-number="378"></td>
        <td id="LC378" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> radius_mask <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L379" class="blob-num js-line-number" data-line-number="379"></td>
        <td id="LC379" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> <span class="pl-c1">self</span>.filename.replace(<span class="pl-s"><span class="pl-pds">&#39;</span>.fits<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>-cones-fft-masked.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L380" class="blob-num js-line-number" data-line-number="380"></td>
        <td id="LC380" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L381" class="blob-num js-line-number" data-line-number="381"></td>
        <td id="LC381" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> <span class="pl-c1">self</span>.filename.replace(<span class="pl-s"><span class="pl-pds">&#39;</span>.fits<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>-cones-fft.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L382" class="blob-num js-line-number" data-line-number="382"></td>
        <td id="LC382" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L383" class="blob-num js-line-number" data-line-number="383"></td>
        <td id="LC383" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> radius_mask <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L384" class="blob-num js-line-number" data-line-number="384"></td>
        <td id="LC384" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>cones-fft-masked.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L385" class="blob-num js-line-number" data-line-number="385"></td>
        <td id="LC385" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L386" class="blob-num js-line-number" data-line-number="386"></td>
        <td id="LC386" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>cones-fft.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L387" class="blob-num js-line-number" data-line-number="387"></td>
        <td id="LC387" class="blob-code blob-code-inner js-file-line">            plt.savefig(output_filename, <span class="pl-v">bbox_inches</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>tight<span class="pl-pds">&#39;</span></span>, <span class="pl-v">pad_inches</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L388" class="blob-num js-line-number" data-line-number="388"></td>
        <td id="LC388" class="blob-code blob-code-inner js-file-line">            plt.close(fig)</td>
      </tr>
      <tr>
        <td id="L389" class="blob-num js-line-number" data-line-number="389"></td>
        <td id="LC389" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(output_filename, <span class="pl-s"><span class="pl-pds">&#39;</span>saved.<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L390" class="blob-num js-line-number" data-line-number="390"></td>
        <td id="LC390" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L391" class="blob-num js-line-number" data-line-number="391"></td>
        <td id="LC391" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_fft_mask</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">radius</span><span class="pl-k">=</span><span class="pl-c1">3</span>):</td>
      </tr>
      <tr>
        <td id="L392" class="blob-num js-line-number" data-line-number="392"></td>
        <td id="LC392" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L393" class="blob-num js-line-number" data-line-number="393"></td>
        <td id="LC393" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        function creating a mask to keep only pixels closer to peaks than radius.</span></td>
      </tr>
      <tr>
        <td id="L394" class="blob-num js-line-number" data-line-number="394"></td>
        <td id="LC394" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param radius: radius in pixels of circular mask around each peak.</span></td>
      </tr>
      <tr>
        <td id="L395" class="blob-num js-line-number" data-line-number="395"></td>
        <td id="LC395" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :return: the mask to be applied on fft (image of same size as self.fft_image_cones).</span></td>
      </tr>
      <tr>
        <td id="L396" class="blob-num js-line-number" data-line-number="396"></td>
        <td id="LC396" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L397" class="blob-num js-line-number" data-line-number="397"></td>
        <td id="LC397" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">self</span>.ks <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L398" class="blob-num js-line-number" data-line-number="398"></td>
        <td id="LC398" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.get_cones_separation_reciprocal()</td>
      </tr>
      <tr>
        <td id="L399" class="blob-num js-line-number" data-line-number="399"></td>
        <td id="LC399" class="blob-code blob-code-inner js-file-line">        center <span class="pl-k">=</span> (np.array((<span class="pl-c1">self</span>.fft_image_cones.shape[::<span class="pl-k">-</span><span class="pl-c1">1</span>])) <span class="pl-k">-</span> <span class="pl-c1">1</span>)<span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L400" class="blob-num js-line-number" data-line-number="400"></td>
        <td id="LC400" class="blob-code blob-code-inner js-file-line">        mask <span class="pl-k">=</span> make_repetitive_mask(<span class="pl-c1">self</span>.fft_image_cones.shape,</td>
      </tr>
      <tr>
        <td id="L401" class="blob-num js-line-number" data-line-number="401"></td>
        <td id="LC401" class="blob-code blob-code-inner js-file-line">                                    radius, <span class="pl-c1">self</span>.ks[<span class="pl-c1">0</span>, :], <span class="pl-c1">self</span>.ks[<span class="pl-c1">1</span>, :], center, <span class="pl-v">nrepetition</span><span class="pl-k">=</span><span class="pl-c1">100</span>)</td>
      </tr>
      <tr>
        <td id="L402" class="blob-num js-line-number" data-line-number="402"></td>
        <td id="LC402" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> nk1 <span class="pl-k">in</span> <span class="pl-c1">range</span> (<span class="pl-k">-</span><span class="pl-c1">50</span>, <span class="pl-c1">50</span>):</td>
      </tr>
      <tr>
        <td id="L403" class="blob-num js-line-number" data-line-number="403"></td>
        <td id="LC403" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> nk2 <span class="pl-k">in</span> <span class="pl-c1">range</span> (<span class="pl-k">-</span><span class="pl-c1">50</span>, <span class="pl-c1">50</span>):</td>
      </tr>
      <tr>
        <td id="L404" class="blob-num js-line-number" data-line-number="404"></td>
        <td id="LC404" class="blob-code blob-code-inner js-file-line">                center_circle <span class="pl-k">=</span> center <span class="pl-k">+</span> nk1 <span class="pl-k">*</span> <span class="pl-c1">self</span>.ks[<span class="pl-c1">0</span>, :] <span class="pl-k">+</span> nk2 <span class="pl-k">*</span> <span class="pl-c1">self</span>.ks[<span class="pl-c1">1</span>, :]</td>
      </tr>
      <tr>
        <td id="L405" class="blob-num js-line-number" data-line-number="405"></td>
        <td id="LC405" class="blob-code blob-code-inner js-file-line">                mask <span class="pl-k">=</span> set_circle(mask, <span class="pl-v">center</span><span class="pl-k">=</span>center_circle.reshape((<span class="pl-c1">2</span>,)), <span class="pl-v">radius</span><span class="pl-k">=</span>radius, <span class="pl-v">value</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L406" class="blob-num js-line-number" data-line-number="406"></td>
        <td id="LC406" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> np.fft.ifftshift(mask)</td>
      </tr>
      <tr>
        <td id="L407" class="blob-num js-line-number" data-line-number="407"></td>
        <td id="LC407" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L408" class="blob-num js-line-number" data-line-number="408"></td>
        <td id="LC408" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_cones_separation_reciprocal</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">output_dir</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L409" class="blob-num js-line-number" data-line-number="409"></td>
        <td id="LC409" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L410" class="blob-num js-line-number" data-line-number="410"></td>
        <td id="LC410" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Function to calculate the distance between 2 neighbour pixels (as vectors) and 3 consecutive radius.</span></td>
      </tr>
      <tr>
        <td id="L411" class="blob-num js-line-number" data-line-number="411"></td>
        <td id="LC411" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        The distances are stored in self.v[12]_lattice and the radius are stored in self.r[123]</span></td>
      </tr>
      <tr>
        <td id="L412" class="blob-num js-line-number" data-line-number="412"></td>
        <td id="LC412" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param output_dir: optional directory where to put the auto-correlation of the FFT image.</span></td>
      </tr>
      <tr>
        <td id="L413" class="blob-num js-line-number" data-line-number="413"></td>
        <td id="LC413" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L414" class="blob-num js-line-number" data-line-number="414"></td>
        <td id="LC414" class="blob-code blob-code-inner js-file-line">        fft_image_cones_shifted <span class="pl-k">=</span> signal.convolve2d(np.abs(<span class="pl-c1">self</span>.fft_image_cones), high_pass_filter_77,</td>
      </tr>
      <tr>
        <td id="L415" class="blob-num js-line-number" data-line-number="415"></td>
        <td id="LC415" class="blob-code blob-code-inner js-file-line">                                                    <span class="pl-v">mode</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>same<span class="pl-pds">&#39;</span></span>, <span class="pl-v">boundary</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>wrap<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L416" class="blob-num js-line-number" data-line-number="416"></td>
        <td id="LC416" class="blob-code blob-code-inner js-file-line">        fft_image_cones_shifted <span class="pl-k">=</span> np.fft.fftshift(fft_image_cones_shifted)</td>
      </tr>
      <tr>
        <td id="L417" class="blob-num js-line-number" data-line-number="417"></td>
        <td id="LC417" class="blob-code blob-code-inner js-file-line">        fft_image_cones_shifted[fft_image_cones_shifted <span class="pl-k">&lt;</span> <span class="pl-c1">0</span>] <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L418" class="blob-num js-line-number" data-line-number="418"></td>
        <td id="LC418" class="blob-code blob-code-inner js-file-line">        ks_base, auto_correlation_saved, center_peaks <span class="pl-k">=</span> get_peaks_separation(fft_image_cones_shifted, <span class="pl-v">crop_range</span><span class="pl-k">=</span><span class="pl-c1">802</span>)</td>
      </tr>
      <tr>
        <td id="L419" class="blob-num js-line-number" data-line-number="419"></td>
        <td id="LC419" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L420" class="blob-num js-line-number" data-line-number="420"></td>
        <td id="LC420" class="blob-code blob-code-inner js-file-line">            plt.ioff()</td>
      </tr>
      <tr>
        <td id="L421" class="blob-num js-line-number" data-line-number="421"></td>
        <td id="LC421" class="blob-code blob-code-inner js-file-line">            fig <span class="pl-k">=</span> plt.figure()</td>
      </tr>
      <tr>
        <td id="L422" class="blob-num js-line-number" data-line-number="422"></td>
        <td id="LC422" class="blob-code blob-code-inner js-file-line">            ax <span class="pl-k">=</span> fig.add_subplot(<span class="pl-c1">111</span>)</td>
      </tr>
      <tr>
        <td id="L423" class="blob-num js-line-number" data-line-number="423"></td>
        <td id="LC423" class="blob-code blob-code-inner js-file-line">            plt.imshow(auto_correlation_saved, <span class="pl-v">cmap</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>gray<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L424" class="blob-num js-line-number" data-line-number="424"></td>
        <td id="LC424" class="blob-code blob-code-inner js-file-line">            plt.plot(center_peaks[:, <span class="pl-c1">0</span>], center_peaks[:, <span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>y+<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L425" class="blob-num js-line-number" data-line-number="425"></td>
        <td id="LC425" class="blob-code blob-code-inner js-file-line">            plt.xlim((<span class="pl-c1">0</span>, auto_correlation_saved.shape[<span class="pl-c1">1</span>]))</td>
      </tr>
      <tr>
        <td id="L426" class="blob-num js-line-number" data-line-number="426"></td>
        <td id="LC426" class="blob-code blob-code-inner js-file-line">            plt.ylim((<span class="pl-c1">0</span>, auto_correlation_saved.shape[<span class="pl-c1">0</span>]))</td>
      </tr>
      <tr>
        <td id="L427" class="blob-num js-line-number" data-line-number="427"></td>
        <td id="LC427" class="blob-code blob-code-inner js-file-line">            k1_points <span class="pl-k">=</span> np.array((np.zeros((<span class="pl-c1">2</span>,)), ks_base[<span class="pl-c1">0</span>, :])) <span class="pl-k">+</span> center_peaks[<span class="pl-c1">0</span>, :]</td>
      </tr>
      <tr>
        <td id="L428" class="blob-num js-line-number" data-line-number="428"></td>
        <td id="LC428" class="blob-code blob-code-inner js-file-line">            k2_points <span class="pl-k">=</span> np.array((np.zeros((<span class="pl-c1">2</span>,)), ks_base[<span class="pl-c1">1</span>, :])) <span class="pl-k">+</span> center_peaks[<span class="pl-c1">0</span>, :]</td>
      </tr>
      <tr>
        <td id="L429" class="blob-num js-line-number" data-line-number="429"></td>
        <td id="LC429" class="blob-code blob-code-inner js-file-line">            plt.plot(k1_points[:, <span class="pl-c1">0</span>], k1_points[:, <span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>b-<span class="pl-pds">&#39;</span></span>, <span class="pl-v">linewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L430" class="blob-num js-line-number" data-line-number="430"></td>
        <td id="LC430" class="blob-code blob-code-inner js-file-line">            plt.plot(k2_points[:, <span class="pl-c1">0</span>], k2_points[:, <span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>r-<span class="pl-pds">&#39;</span></span>, <span class="pl-v">linewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L431" class="blob-num js-line-number" data-line-number="431"></td>
        <td id="LC431" class="blob-code blob-code-inner js-file-line">            plt.grid(<span class="pl-c1">None</span>)</td>
      </tr>
      <tr>
        <td id="L432" class="blob-num js-line-number" data-line-number="432"></td>
        <td id="LC432" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> <span class="pl-c1">self</span>.filename <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L433" class="blob-num js-line-number" data-line-number="433"></td>
        <td id="LC433" class="blob-code blob-code-inner js-file-line">                output_filename <span class="pl-k">=</span> <span class="pl-c1">self</span>.filename.replace(<span class="pl-s"><span class="pl-pds">&#39;</span>.fits<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>-cones-sep-reciprocal.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L434" class="blob-num js-line-number" data-line-number="434"></td>
        <td id="LC434" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L435" class="blob-num js-line-number" data-line-number="435"></td>
        <td id="LC435" class="blob-code blob-code-inner js-file-line">                output_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>cones-sep-reciprocal.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L436" class="blob-num js-line-number" data-line-number="436"></td>
        <td id="LC436" class="blob-code blob-code-inner js-file-line">            plt.savefig(output_filename, <span class="pl-v">bbox_inches</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>tight<span class="pl-pds">&#39;</span></span>, <span class="pl-v">pad_inches</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L437" class="blob-num js-line-number" data-line-number="437"></td>
        <td id="LC437" class="blob-code blob-code-inner js-file-line">            plt.close(fig)</td>
      </tr>
      <tr>
        <td id="L438" class="blob-num js-line-number" data-line-number="438"></td>
        <td id="LC438" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(output_filename, <span class="pl-s"><span class="pl-pds">&#39;</span>saved.<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L439" class="blob-num js-line-number" data-line-number="439"></td>
        <td id="LC439" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ks <span class="pl-k">=</span> ks_base</td>
      </tr>
      <tr>
        <td id="L440" class="blob-num js-line-number" data-line-number="440"></td>
        <td id="LC440" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.v1_lattice, <span class="pl-c1">self</span>.v2_lattice <span class="pl-k">=</span> reciprocal_to_lattice_space(<span class="pl-c1">self</span>.ks[<span class="pl-c1">0</span>, :], <span class="pl-c1">self</span>.ks[<span class="pl-c1">1</span>, :], <span class="pl-c1">self</span>.fft_image_cones.shape[::<span class="pl-k">-</span><span class="pl-c1">1</span>])</td>
      </tr>
      <tr>
        <td id="L441" class="blob-num js-line-number" data-line-number="441"></td>
        <td id="LC441" class="blob-code blob-code-inner js-file-line">        r3 <span class="pl-k">=</span> (<span class="pl-c1">self</span>.v1_lattice <span class="pl-k">+</span> <span class="pl-c1">self</span>.v2_lattice) <span class="pl-k">/</span> <span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L442" class="blob-num js-line-number" data-line-number="442"></td>
        <td id="LC442" class="blob-code blob-code-inner js-file-line">        r1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.v1_lattice <span class="pl-k">-</span> r3</td>
      </tr>
      <tr>
        <td id="L443" class="blob-num js-line-number" data-line-number="443"></td>
        <td id="LC443" class="blob-code blob-code-inner js-file-line">        r2 <span class="pl-k">=</span> <span class="pl-c1">self</span>.v2_lattice <span class="pl-k">-</span> r3</td>
      </tr>
      <tr>
        <td id="L444" class="blob-num js-line-number" data-line-number="444"></td>
        <td id="LC444" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.r1, <span class="pl-c1">self</span>.r2 <span class="pl-k">=</span> get_consecutive_hex_radius(r1, r2)</td>
      </tr>
      <tr>
        <td id="L445" class="blob-num js-line-number" data-line-number="445"></td>
        <td id="LC445" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.r3 <span class="pl-k">=</span> <span class="pl-c1">self</span>.r2 <span class="pl-k">-</span> <span class="pl-c1">self</span>.r1</td>
      </tr>
      <tr>
        <td id="L446" class="blob-num js-line-number" data-line-number="446"></td>
        <td id="LC446" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>r1=<span class="pl-pds">&quot;</span></span>, <span class="pl-c1">self</span>.r1, <span class="pl-s"><span class="pl-pds">&quot;</span>|r1|=<span class="pl-pds">&quot;</span></span>, np.abs(<span class="pl-c1">self</span>.r1[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.r1[<span class="pl-c1">1</span>]))</td>
      </tr>
      <tr>
        <td id="L447" class="blob-num js-line-number" data-line-number="447"></td>
        <td id="LC447" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>r2=<span class="pl-pds">&quot;</span></span>, <span class="pl-c1">self</span>.r2, <span class="pl-s"><span class="pl-pds">&quot;</span>|r2|=<span class="pl-pds">&quot;</span></span>, np.abs(<span class="pl-c1">self</span>.r2[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.r2[<span class="pl-c1">1</span>]))</td>
      </tr>
      <tr>
        <td id="L448" class="blob-num js-line-number" data-line-number="448"></td>
        <td id="LC448" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>r3=<span class="pl-pds">&quot;</span></span>, <span class="pl-c1">self</span>.r3, <span class="pl-s"><span class="pl-pds">&quot;</span>|r3|=<span class="pl-pds">&quot;</span></span>, np.abs(<span class="pl-c1">self</span>.r3[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.r3[<span class="pl-c1">1</span>]))</td>
      </tr>
      <tr>
        <td id="L449" class="blob-num js-line-number" data-line-number="449"></td>
        <td id="LC449" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L450" class="blob-num js-line-number" data-line-number="450"></td>
        <td id="LC450" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">plot_cones_presence</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">radius_mask</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">output_dir</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L451" class="blob-num js-line-number" data-line-number="451"></td>
        <td id="LC451" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L452" class="blob-num js-line-number" data-line-number="452"></td>
        <td id="LC452" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Do convolution of cone image on filtered lid CCD image, then fit peaks positions.</span></td>
      </tr>
      <tr>
        <td id="L453" class="blob-num js-line-number" data-line-number="453"></td>
        <td id="LC453" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Find the best orientation and position for the camera geometry to match the peaks.</span></td>
      </tr>
      <tr>
        <td id="L454" class="blob-num js-line-number" data-line-number="454"></td>
        <td id="LC454" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Finally refit the distance between pixels only taking into account the matching peaks.</span></td>
      </tr>
      <tr>
        <td id="L455" class="blob-num js-line-number" data-line-number="455"></td>
        <td id="LC455" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param radius_mask: radius of the mask used to keep only relevant peaks in the convultion.</span></td>
      </tr>
      <tr>
        <td id="L456" class="blob-num js-line-number" data-line-number="456"></td>
        <td id="LC456" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        :param output_dir: optional directory where to put the resulting images.</span></td>
      </tr>
      <tr>
        <td id="L457" class="blob-num js-line-number" data-line-number="457"></td>
        <td id="LC457" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        If None (default) the image is displayed instead of being saved to a file.</span></td>
      </tr>
      <tr>
        <td id="L458" class="blob-num js-line-number" data-line-number="458"></td>
        <td id="LC458" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L459" class="blob-num js-line-number" data-line-number="459"></td>
        <td id="LC459" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">self</span>.image_cone <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L460" class="blob-num js-line-number" data-line-number="460"></td>
        <td id="LC460" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> radius_mask <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L461" class="blob-num js-line-number" data-line-number="461"></td>
        <td id="LC461" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">raise</span> <span class="pl-c1">AttributeError</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>cone_image was not calculated and no radius mask given<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L462" class="blob-num js-line-number" data-line-number="462"></td>
        <td id="LC462" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.get_cone(<span class="pl-v">radius_mask</span><span class="pl-k">=</span>radius_mask, <span class="pl-v">output_dir</span><span class="pl-k">=</span>output_dir)</td>
      </tr>
      <tr>
        <td id="L463" class="blob-num js-line-number" data-line-number="463"></td>
        <td id="LC463" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>find peaks<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L464" class="blob-num js-line-number" data-line-number="464"></td>
        <td id="LC464" class="blob-code blob-code-inner js-file-line">        cone_presence <span class="pl-k">=</span> signal.fftconvolve(<span class="pl-c1">self</span>.image_cones, <span class="pl-c1">self</span>.image_cone, <span class="pl-v">mode</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>same<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L465" class="blob-num js-line-number" data-line-number="465"></td>
        <td id="LC465" class="blob-code blob-code-inner js-file-line">        cone_presence <span class="pl-k">=</span> signal.fftconvolve(cone_presence, high_pass_filter_2525, <span class="pl-v">mode</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>same<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L466" class="blob-num js-line-number" data-line-number="466"></td>
        <td id="LC466" class="blob-code blob-code-inner js-file-line">        cone_presence[cone_presence <span class="pl-k">&lt;</span> <span class="pl-c1">0</span>]<span class="pl-k">=</span><span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L467" class="blob-num js-line-number" data-line-number="467"></td>
        <td id="LC467" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span>cone_presence = cv2.matchTemplate(self.image_cones.astype(np.float32), self.image_cone.astype(np.float32), cv2.TM_CCOEFF_NORMED)</span></td>
      </tr>
      <tr>
        <td id="L468" class="blob-num js-line-number" data-line-number="468"></td>
        <td id="LC468" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L469" class="blob-num js-line-number" data-line-number="469"></td>
        <td id="LC469" class="blob-code blob-code-inner js-file-line">            plt.ioff()</td>
      </tr>
      <tr>
        <td id="L470" class="blob-num js-line-number" data-line-number="470"></td>
        <td id="LC470" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L471" class="blob-num js-line-number" data-line-number="471"></td>
        <td id="LC471" class="blob-code blob-code-inner js-file-line">            plt.ion()</td>
      </tr>
      <tr>
        <td id="L472" class="blob-num js-line-number" data-line-number="472"></td>
        <td id="LC472" class="blob-code blob-code-inner js-file-line">        fig <span class="pl-k">=</span> plt.figure()</td>
      </tr>
      <tr>
        <td id="L473" class="blob-num js-line-number" data-line-number="473"></td>
        <td id="LC473" class="blob-code blob-code-inner js-file-line">        ax <span class="pl-k">=</span> plt.gca()</td>
      </tr>
      <tr>
        <td id="L474" class="blob-num js-line-number" data-line-number="474"></td>
        <td id="LC474" class="blob-code blob-code-inner js-file-line">        plt.imshow(cone_presence, <span class="pl-v">cmap</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>gray<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L475" class="blob-num js-line-number" data-line-number="475"></td>
        <td id="LC475" class="blob-code blob-code-inner js-file-line">        plt.grid(<span class="pl-c1">None</span>)</td>
      </tr>
      <tr>
        <td id="L476" class="blob-num js-line-number" data-line-number="476"></td>
        <td id="LC476" class="blob-code blob-code-inner js-file-line">        plt.axis(<span class="pl-s"><span class="pl-pds">&#39;</span>off<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L477" class="blob-num js-line-number" data-line-number="477"></td>
        <td id="LC477" class="blob-code blob-code-inner js-file-line">        ax.get_xaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L478" class="blob-num js-line-number" data-line-number="478"></td>
        <td id="LC478" class="blob-code blob-code-inner js-file-line">        ax.get_yaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L479" class="blob-num js-line-number" data-line-number="479"></td>
        <td id="LC479" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L480" class="blob-num js-line-number" data-line-number="480"></td>
        <td id="LC480" class="blob-code blob-code-inner js-file-line">            plt.show()</td>
      </tr>
      <tr>
        <td id="L481" class="blob-num js-line-number" data-line-number="481"></td>
        <td id="LC481" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L482" class="blob-num js-line-number" data-line-number="482"></td>
        <td id="LC482" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> <span class="pl-c1">self</span>.filename <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L483" class="blob-num js-line-number" data-line-number="483"></td>
        <td id="LC483" class="blob-code blob-code-inner js-file-line">                output_filename <span class="pl-k">=</span> <span class="pl-c1">self</span>.filename.replace(<span class="pl-s"><span class="pl-pds">&#39;</span>.fits<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>-cones-presence.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L484" class="blob-num js-line-number" data-line-number="484"></td>
        <td id="LC484" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L485" class="blob-num js-line-number" data-line-number="485"></td>
        <td id="LC485" class="blob-code blob-code-inner js-file-line">                output_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>cones-presence.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L486" class="blob-num js-line-number" data-line-number="486"></td>
        <td id="LC486" class="blob-code blob-code-inner js-file-line">            plt.savefig(output_filename, <span class="pl-v">bbox_inches</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>tight<span class="pl-pds">&#39;</span></span>, <span class="pl-v">pad_inches</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L487" class="blob-num js-line-number" data-line-number="487"></td>
        <td id="LC487" class="blob-code blob-code-inner js-file-line">            plt.close(fig)</td>
      </tr>
      <tr>
        <td id="L488" class="blob-num js-line-number" data-line-number="488"></td>
        <td id="LC488" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(output_filename, <span class="pl-s"><span class="pl-pds">&#39;</span>saved.<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L489" class="blob-num js-line-number" data-line-number="489"></td>
        <td id="LC489" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> radius_mask <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L490" class="blob-num js-line-number" data-line-number="490"></td>
        <td id="LC490" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>find distance between pixels<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L491" class="blob-num js-line-number" data-line-number="491"></td>
        <td id="LC491" class="blob-code blob-code-inner js-file-line">            cone_presence_fft_shifted <span class="pl-k">=</span> np.fft.fftshift(np.abs(np.fft.fft2(cone_presence)))</td>
      </tr>
      <tr>
        <td id="L492" class="blob-num js-line-number" data-line-number="492"></td>
        <td id="LC492" class="blob-code blob-code-inner js-file-line">            cone_presence_fft_shifted <span class="pl-k">=</span> signal.fftconvolve(cone_presence_fft_shifted, high_pass_filter_77, <span class="pl-v">mode</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>same<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L493" class="blob-num js-line-number" data-line-number="493"></td>
        <td id="LC493" class="blob-code blob-code-inner js-file-line">            ks, auto_correlation, center_peaks <span class="pl-k">=</span> get_peaks_separation(cone_presence_fft_shifted,</td>
      </tr>
      <tr>
        <td id="L494" class="blob-num js-line-number" data-line-number="494"></td>
        <td id="LC494" class="blob-code blob-code-inner js-file-line">                                                                      <span class="pl-v">center</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-v">crop_range</span><span class="pl-k">=</span><span class="pl-c1">800</span>, <span class="pl-v">radius_removed</span><span class="pl-k">=</span><span class="pl-c1">20</span>)</td>
      </tr>
      <tr>
        <td id="L495" class="blob-num js-line-number" data-line-number="495"></td>
        <td id="LC495" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">del</span> cone_presence_fft_shifted</td>
      </tr>
      <tr>
        <td id="L496" class="blob-num js-line-number" data-line-number="496"></td>
        <td id="LC496" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L497" class="blob-num js-line-number" data-line-number="497"></td>
        <td id="LC497" class="blob-code blob-code-inner js-file-line">                plt.ioff()</td>
      </tr>
      <tr>
        <td id="L498" class="blob-num js-line-number" data-line-number="498"></td>
        <td id="LC498" class="blob-code blob-code-inner js-file-line">                fig <span class="pl-k">=</span> plt.figure()</td>
      </tr>
      <tr>
        <td id="L499" class="blob-num js-line-number" data-line-number="499"></td>
        <td id="LC499" class="blob-code blob-code-inner js-file-line">                ax <span class="pl-k">=</span> fig.add_subplot(<span class="pl-c1">111</span>)</td>
      </tr>
      <tr>
        <td id="L500" class="blob-num js-line-number" data-line-number="500"></td>
        <td id="LC500" class="blob-code blob-code-inner js-file-line">                plt.imshow(auto_correlation, <span class="pl-v">cmap</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>gray<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L501" class="blob-num js-line-number" data-line-number="501"></td>
        <td id="LC501" class="blob-code blob-code-inner js-file-line">                plt.plot(center_peaks[:, <span class="pl-c1">0</span>], center_peaks[:, <span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>y+<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L502" class="blob-num js-line-number" data-line-number="502"></td>
        <td id="LC502" class="blob-code blob-code-inner js-file-line">                plt.xlim((<span class="pl-c1">0</span>, auto_correlation.shape[<span class="pl-c1">1</span>]))</td>
      </tr>
      <tr>
        <td id="L503" class="blob-num js-line-number" data-line-number="503"></td>
        <td id="LC503" class="blob-code blob-code-inner js-file-line">                plt.ylim((<span class="pl-c1">0</span>, auto_correlation.shape[<span class="pl-c1">0</span>]))</td>
      </tr>
      <tr>
        <td id="L504" class="blob-num js-line-number" data-line-number="504"></td>
        <td id="LC504" class="blob-code blob-code-inner js-file-line">                k1_points <span class="pl-k">=</span> np.array((<span class="pl-c1">0</span> <span class="pl-k">*</span> ks[<span class="pl-c1">0</span>, :], ks[<span class="pl-c1">0</span>, :])) <span class="pl-k">+</span> center_peaks[<span class="pl-c1">0</span>, :]</td>
      </tr>
      <tr>
        <td id="L505" class="blob-num js-line-number" data-line-number="505"></td>
        <td id="LC505" class="blob-code blob-code-inner js-file-line">                k2_points <span class="pl-k">=</span> np.array((<span class="pl-c1">0</span> <span class="pl-k">*</span> ks[<span class="pl-c1">0</span>, :], ks[<span class="pl-c1">1</span>, :])) <span class="pl-k">+</span> center_peaks[<span class="pl-c1">0</span>, :]</td>
      </tr>
      <tr>
        <td id="L506" class="blob-num js-line-number" data-line-number="506"></td>
        <td id="LC506" class="blob-code blob-code-inner js-file-line">                plt.plot(k1_points[:, <span class="pl-c1">0</span>], k1_points[:, <span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>b-<span class="pl-pds">&#39;</span></span>, <span class="pl-v">linewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L507" class="blob-num js-line-number" data-line-number="507"></td>
        <td id="LC507" class="blob-code blob-code-inner js-file-line">                plt.plot(k2_points[:, <span class="pl-c1">0</span>], k2_points[:, <span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>r-<span class="pl-pds">&#39;</span></span>, <span class="pl-v">linewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L508" class="blob-num js-line-number" data-line-number="508"></td>
        <td id="LC508" class="blob-code blob-code-inner js-file-line">                plt.grid(<span class="pl-c1">None</span>)</td>
      </tr>
      <tr>
        <td id="L509" class="blob-num js-line-number" data-line-number="509"></td>
        <td id="LC509" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> <span class="pl-c1">self</span>.filename <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L510" class="blob-num js-line-number" data-line-number="510"></td>
        <td id="LC510" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> <span class="pl-c1">self</span>.filename.replace(<span class="pl-s"><span class="pl-pds">&#39;</span>.fits<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>-cones-sep.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L511" class="blob-num js-line-number" data-line-number="511"></td>
        <td id="LC511" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L512" class="blob-num js-line-number" data-line-number="512"></td>
        <td id="LC512" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>cones-sep.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L513" class="blob-num js-line-number" data-line-number="513"></td>
        <td id="LC513" class="blob-code blob-code-inner js-file-line">                plt.savefig(output_filename, <span class="pl-v">bbox_inches</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>tight<span class="pl-pds">&#39;</span></span>, <span class="pl-v">pad_inches</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L514" class="blob-num js-line-number" data-line-number="514"></td>
        <td id="LC514" class="blob-code blob-code-inner js-file-line">                plt.close(fig)</td>
      </tr>
      <tr>
        <td id="L515" class="blob-num js-line-number" data-line-number="515"></td>
        <td id="LC515" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">print</span>(output_filename, <span class="pl-s"><span class="pl-pds">&#39;</span>saved.<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L516" class="blob-num js-line-number" data-line-number="516"></td>
        <td id="LC516" class="blob-code blob-code-inner js-file-line">            v1, v2 <span class="pl-k">=</span> reciprocal_to_lattice_space(ks[<span class="pl-c1">0</span>, :],  ks[<span class="pl-c1">1</span>, :], cone_presence.shape[::<span class="pl-k">-</span><span class="pl-c1">1</span>])</td>
      </tr>
      <tr>
        <td id="L517" class="blob-num js-line-number" data-line-number="517"></td>
        <td id="LC517" class="blob-code blob-code-inner js-file-line">            v1, v2 <span class="pl-k">=</span> get_consecutive_hex_radius(v1, v2)</td>
      </tr>
      <tr>
        <td id="L518" class="blob-num js-line-number" data-line-number="518"></td>
        <td id="LC518" class="blob-code blob-code-inner js-file-line">            v3 <span class="pl-k">=</span> v2 <span class="pl-k">-</span> v1</td>
      </tr>
      <tr>
        <td id="L519" class="blob-num js-line-number" data-line-number="519"></td>
        <td id="LC519" class="blob-code blob-code-inner js-file-line">            v1_length <span class="pl-k">=</span> np.abs(v1[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> v1[<span class="pl-c1">1</span>])</td>
      </tr>
      <tr>
        <td id="L520" class="blob-num js-line-number" data-line-number="520"></td>
        <td id="LC520" class="blob-code blob-code-inner js-file-line">            v2_length <span class="pl-k">=</span> np.abs(v2[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> v2[<span class="pl-c1">1</span>])</td>
      </tr>
      <tr>
        <td id="L521" class="blob-num js-line-number" data-line-number="521"></td>
        <td id="LC521" class="blob-code blob-code-inner js-file-line">            v3_length <span class="pl-k">=</span> np.abs(v3[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> v3[<span class="pl-c1">1</span>])</td>
      </tr>
      <tr>
        <td id="L522" class="blob-num js-line-number" data-line-number="522"></td>
        <td id="LC522" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>v1=<span class="pl-pds">&quot;</span></span>, v1, <span class="pl-s"><span class="pl-pds">&quot;</span>|v1|=<span class="pl-pds">&quot;</span></span>, v1_length)</td>
      </tr>
      <tr>
        <td id="L523" class="blob-num js-line-number" data-line-number="523"></td>
        <td id="LC523" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>v2=<span class="pl-pds">&quot;</span></span>, v2, <span class="pl-s"><span class="pl-pds">&quot;</span>|v2|=<span class="pl-pds">&quot;</span></span>, v2_length)</td>
      </tr>
      <tr>
        <td id="L524" class="blob-num js-line-number" data-line-number="524"></td>
        <td id="LC524" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>v3=<span class="pl-pds">&quot;</span></span>, v3, <span class="pl-s"><span class="pl-pds">&quot;</span>|v3|=<span class="pl-pds">&quot;</span></span>, v3_length)</td>
      </tr>
      <tr>
        <td id="L525" class="blob-num js-line-number" data-line-number="525"></td>
        <td id="LC525" class="blob-code blob-code-inner js-file-line">            pixels_fit_px <span class="pl-k">=</span> []</td>
      </tr>
      <tr>
        <td id="L526" class="blob-num js-line-number" data-line-number="526"></td>
        <td id="LC526" class="blob-code blob-code-inner js-file-line">            pixels_fit_sigma <span class="pl-k">=</span> []</td>
      </tr>
      <tr>
        <td id="L527" class="blob-num js-line-number" data-line-number="527"></td>
        <td id="LC527" class="blob-code blob-code-inner js-file-line">            nfail<span class="pl-k">=</span><span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L528" class="blob-num js-line-number" data-line-number="528"></td>
        <td id="LC528" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>fit all peaks<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L529" class="blob-num js-line-number" data-line-number="529"></td>
        <td id="LC529" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> i1 <span class="pl-k">in</span> <span class="pl-c1">range</span>(<span class="pl-k">-</span><span class="pl-c1">100</span>, <span class="pl-c1">100</span>):</td>
      </tr>
      <tr>
        <td id="L530" class="blob-num js-line-number" data-line-number="530"></td>
        <td id="LC530" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">for</span> i2 <span class="pl-k">in</span> <span class="pl-c1">range</span>(<span class="pl-k">-</span><span class="pl-c1">100</span>, <span class="pl-c1">100</span>):</td>
      </tr>
      <tr>
        <td id="L531" class="blob-num js-line-number" data-line-number="531"></td>
        <td id="LC531" class="blob-code blob-code-inner js-file-line">                    peak_pos_aprox <span class="pl-k">=</span> <span class="pl-c1">self</span>.center_fitted <span class="pl-k">+</span> i1 <span class="pl-k">*</span> v1 <span class="pl-k">+</span> i2 <span class="pl-k">*</span> v2</td>
      </tr>
      <tr>
        <td id="L532" class="blob-num js-line-number" data-line-number="532"></td>
        <td id="LC532" class="blob-code blob-code-inner js-file-line">                    crop_px1 <span class="pl-k">=</span> np.floor(peak_pos_aprox <span class="pl-k">-</span> radius_mask)</td>
      </tr>
      <tr>
        <td id="L533" class="blob-num js-line-number" data-line-number="533"></td>
        <td id="LC533" class="blob-code blob-code-inner js-file-line">                    crop_px1 <span class="pl-k">=</span> np.maximum(crop_px1, (<span class="pl-c1">0</span>, <span class="pl-c1">0</span>))</td>
      </tr>
      <tr>
        <td id="L534" class="blob-num js-line-number" data-line-number="534"></td>
        <td id="LC534" class="blob-code blob-code-inner js-file-line">                    crop_px1 <span class="pl-k">=</span> np.minimum(crop_px1, (cone_presence.shape[<span class="pl-c1">1</span>] <span class="pl-k">-</span> <span class="pl-c1">1</span>, cone_presence.shape[<span class="pl-c1">0</span>] <span class="pl-k">-</span> <span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L535" class="blob-num js-line-number" data-line-number="535"></td>
        <td id="LC535" class="blob-code blob-code-inner js-file-line">                    crop_px1 <span class="pl-k">=</span> crop_px1.astype(<span class="pl-c1">int</span>)</td>
      </tr>
      <tr>
        <td id="L536" class="blob-num js-line-number" data-line-number="536"></td>
        <td id="LC536" class="blob-code blob-code-inner js-file-line">                    crop_px2 <span class="pl-k">=</span> np.ceil(peak_pos_aprox <span class="pl-k">+</span> radius_mask)</td>
      </tr>
      <tr>
        <td id="L537" class="blob-num js-line-number" data-line-number="537"></td>
        <td id="LC537" class="blob-code blob-code-inner js-file-line">                    crop_px2 <span class="pl-k">=</span> np.maximum(crop_px2, (<span class="pl-c1">0</span>, <span class="pl-c1">0</span>))</td>
      </tr>
      <tr>
        <td id="L538" class="blob-num js-line-number" data-line-number="538"></td>
        <td id="LC538" class="blob-code blob-code-inner js-file-line">                    crop_px2 <span class="pl-k">=</span> np.minimum(crop_px2, (cone_presence.shape[<span class="pl-c1">1</span>] <span class="pl-k">-</span> <span class="pl-c1">1</span>, cone_presence.shape[<span class="pl-c1">0</span>] <span class="pl-k">-</span> <span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L539" class="blob-num js-line-number" data-line-number="539"></td>
        <td id="LC539" class="blob-code blob-code-inner js-file-line">                    crop_px2 <span class="pl-k">=</span> crop_px2.astype(<span class="pl-c1">int</span>)</td>
      </tr>
      <tr>
        <td id="L540" class="blob-num js-line-number" data-line-number="540"></td>
        <td id="LC540" class="blob-code blob-code-inner js-file-line">                    <span class="pl-k">if</span> np.any(crop_px2 <span class="pl-k">-</span> crop_px1 <span class="pl-k">&lt;=</span> np.round(radius_mask)):</td>
      </tr>
      <tr>
        <td id="L541" class="blob-num js-line-number" data-line-number="541"></td>
        <td id="LC541" class="blob-code blob-code-inner js-file-line">                        <span class="pl-k">continue</span></td>
      </tr>
      <tr>
        <td id="L542" class="blob-num js-line-number" data-line-number="542"></td>
        <td id="LC542" class="blob-code blob-code-inner js-file-line">                    crop_center <span class="pl-k">=</span> (crop_px2 <span class="pl-k">-</span> crop_px1 <span class="pl-k">-</span> <span class="pl-c1">1</span>.) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L543" class="blob-num js-line-number" data-line-number="543"></td>
        <td id="LC543" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c"><span class="pl-c">#</span> print(&#39;fit around:&#39;, peak_pos_aprox)</span></td>
      </tr>
      <tr>
        <td id="L544" class="blob-num js-line-number" data-line-number="544"></td>
        <td id="LC544" class="blob-code blob-code-inner js-file-line">                    peak_crop, crop_px1, crop_px2 <span class="pl-k">=</span> crop_image(cone_presence, crop_px1, crop_px2)</td>
      </tr>
      <tr>
        <td id="L545" class="blob-num js-line-number" data-line-number="545"></td>
        <td id="LC545" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c"><span class="pl-c">#</span>cone_presence_filtered[crop_px1[1]:crop_px2[1], crop_px1[0]:crop_px2[0]] = peak_crop</span></td>
      </tr>
      <tr>
        <td id="L546" class="blob-num js-line-number" data-line-number="546"></td>
        <td id="LC546" class="blob-code blob-code-inner js-file-line">                    max_pos_crop <span class="pl-k">=</span> np.argmax(peak_crop)</td>
      </tr>
      <tr>
        <td id="L547" class="blob-num js-line-number" data-line-number="547"></td>
        <td id="LC547" class="blob-code blob-code-inner js-file-line">                    [max_pos_y, max_pos_x] <span class="pl-k">=</span> np.unravel_index(max_pos_crop, peak_crop.shape)</td>
      </tr>
      <tr>
        <td id="L548" class="blob-num js-line-number" data-line-number="548"></td>
        <td id="LC548" class="blob-code blob-code-inner js-file-line">                    init_amplitude <span class="pl-k">=</span> peak_crop[max_pos_y, max_pos_x] <span class="pl-k">-</span> np.min(peak_crop)</td>
      </tr>
      <tr>
        <td id="L549" class="blob-num js-line-number" data-line-number="549"></td>
        <td id="LC549" class="blob-code blob-code-inner js-file-line">                    init_param <span class="pl-k">=</span> (init_amplitude, max_pos_x, max_pos_y, <span class="pl-c1">4</span>, <span class="pl-c1">4</span>, <span class="pl-c1">0</span>, np.min(peak_crop))</td>
      </tr>
      <tr>
        <td id="L550" class="blob-num js-line-number" data-line-number="550"></td>
        <td id="LC550" class="blob-code blob-code-inner js-file-line">                    fit_result, success <span class="pl-k">=</span> FitGauss2D(peak_crop.transpose(), <span class="pl-v">ip</span><span class="pl-k">=</span>init_param)</td>
      </tr>
      <tr>
        <td id="L551" class="blob-num js-line-number" data-line-number="551"></td>
        <td id="LC551" class="blob-code blob-code-inner js-file-line">                    amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg <span class="pl-k">=</span> fit_result</td>
      </tr>
      <tr>
        <td id="L552" class="blob-num js-line-number" data-line-number="552"></td>
        <td id="LC552" class="blob-code blob-code-inner js-file-line">                    <span class="pl-k">if</span> <span class="pl-c1">0</span> <span class="pl-k">&lt;</span> success <span class="pl-k">&lt;=</span> <span class="pl-c1">4</span> <span class="pl-k">and</span> <span class="pl-c1">0</span> <span class="pl-k">&lt;</span> xsigma <span class="pl-k">&lt;</span> <span class="pl-c1">6</span> <span class="pl-k">and</span> <span class="pl-c1">0</span> <span class="pl-k">&lt;</span> ysigma <span class="pl-k">&lt;</span> <span class="pl-c1">6</span> <span class="pl-k">and</span>\</td>
      </tr>
      <tr>
        <td id="L553" class="blob-num js-line-number" data-line-number="553"></td>
        <td id="LC553" class="blob-code blob-code-inner js-file-line">                                            <span class="pl-k">-</span><span class="pl-c1">5</span> <span class="pl-k">&lt;</span> xcenter <span class="pl-k">-</span> crop_center[<span class="pl-c1">0</span>] <span class="pl-k">&lt;</span> <span class="pl-c1">5</span> <span class="pl-k">and</span> <span class="pl-k">-</span><span class="pl-c1">5</span> <span class="pl-k">&lt;</span> ycenter <span class="pl-k">-</span> crop_center[<span class="pl-c1">1</span>] <span class="pl-k">&lt;</span> <span class="pl-c1">5</span>:</td>
      </tr>
      <tr>
        <td id="L554" class="blob-num js-line-number" data-line-number="554"></td>
        <td id="LC554" class="blob-code blob-code-inner js-file-line">                        <span class="pl-c"><span class="pl-c">#</span> print(&#39;SUCCESS: xcenter:&#39;, np.round(xcenter + crop_px1[0], 2),</span></td>
      </tr>
      <tr>
        <td id="L555" class="blob-num js-line-number" data-line-number="555"></td>
        <td id="LC555" class="blob-code blob-code-inner js-file-line">                        <span class="pl-c"><span class="pl-c">#</span>       &#39;ycenter:&#39;, np.round(ycenter + crop_px1[0], 2),</span></td>
      </tr>
      <tr>
        <td id="L556" class="blob-num js-line-number" data-line-number="556"></td>
        <td id="LC556" class="blob-code blob-code-inner js-file-line">                        <span class="pl-c"><span class="pl-c">#</span>       &#39;xsigma:&#39;, np.round(xsigma,2), &#39;ysigma:&#39;, np.round(ysigma,2))</span></td>
      </tr>
      <tr>
        <td id="L557" class="blob-num js-line-number" data-line-number="557"></td>
        <td id="LC557" class="blob-code blob-code-inner js-file-line">                        pixels_fit_px.append(np.array([xcenter, ycenter]) <span class="pl-k">+</span> crop_px1)</td>
      </tr>
      <tr>
        <td id="L558" class="blob-num js-line-number" data-line-number="558"></td>
        <td id="LC558" class="blob-code blob-code-inner js-file-line">                        pixels_fit_sigma.append(np.array([xsigma, ysigma]))</td>
      </tr>
      <tr>
        <td id="L559" class="blob-num js-line-number" data-line-number="559"></td>
        <td id="LC559" class="blob-code blob-code-inner js-file-line">                    <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L560" class="blob-num js-line-number" data-line-number="560"></td>
        <td id="LC560" class="blob-code blob-code-inner js-file-line">                        <span class="pl-c"><span class="pl-c">#</span> print(&#39;FAIL: xcenter:&#39;, np.round(xcenter, 2), &#39;ycenter:&#39;, np.round(ycenter, 2),</span></td>
      </tr>
      <tr>
        <td id="L561" class="blob-num js-line-number" data-line-number="561"></td>
        <td id="LC561" class="blob-code blob-code-inner js-file-line">                        <span class="pl-c"><span class="pl-c">#</span>       &#39;xsigma:&#39;, np.round(xsigma, 2), &#39;ysigma:&#39;, np.round(ysigma, 2))</span></td>
      </tr>
      <tr>
        <td id="L562" class="blob-num js-line-number" data-line-number="562"></td>
        <td id="LC562" class="blob-code blob-code-inner js-file-line">                        nfail <span class="pl-k">+=</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L563" class="blob-num js-line-number" data-line-number="563"></td>
        <td id="LC563" class="blob-code blob-code-inner js-file-line">                    <span class="pl-k">if</span> np.mod(<span class="pl-c1">len</span>(pixels_fit_px)<span class="pl-k">+</span>nfail,<span class="pl-c1">100</span>) <span class="pl-k">==</span> <span class="pl-c1">0</span>:</td>
      </tr>
      <tr>
        <td id="L564" class="blob-num js-line-number" data-line-number="564"></td>
        <td id="LC564" class="blob-code blob-code-inner js-file-line">                        <span class="pl-c1">print</span>(<span class="pl-c1">len</span>(pixels_fit_px)<span class="pl-k">+</span>nfail, <span class="pl-s"><span class="pl-pds">&#39;</span>fits done<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L565" class="blob-num js-line-number" data-line-number="565"></td>
        <td id="LC565" class="blob-code blob-code-inner js-file-line">            pixels_fit_px <span class="pl-k">=</span> np.array(pixels_fit_px)</td>
      </tr>
      <tr>
        <td id="L566" class="blob-num js-line-number" data-line-number="566"></td>
        <td id="LC566" class="blob-code blob-code-inner js-file-line">            pixels_fit_sigma <span class="pl-k">=</span> np.array(pixels_fit_sigma)</td>
      </tr>
      <tr>
        <td id="L567" class="blob-num js-line-number" data-line-number="567"></td>
        <td id="LC567" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>success =<span class="pl-pds">&#39;</span></span>, pixels_fit_px.shape[<span class="pl-c1">0</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>/<span class="pl-pds">&#39;</span></span>, pixels_fit_px.shape[<span class="pl-c1">0</span>] <span class="pl-k">+</span> nfail)</td>
      </tr>
      <tr>
        <td id="L568" class="blob-num js-line-number" data-line-number="568"></td>
        <td id="LC568" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>look for pixels geometry:<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L569" class="blob-num js-line-number" data-line-number="569"></td>
        <td id="LC569" class="blob-code blob-code-inner js-file-line">            best_match_nv1, best_match_nv2 <span class="pl-k">=</span> <span class="pl-c1">0</span>, <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L570" class="blob-num js-line-number" data-line-number="570"></td>
        <td id="LC570" class="blob-code blob-code-inner js-file-line">            best_match_v1, best_match_v2 <span class="pl-k">=</span> v1, v2</td>
      </tr>
      <tr>
        <td id="L571" class="blob-num js-line-number" data-line-number="571"></td>
        <td id="LC571" class="blob-code blob-code-inner js-file-line">            best_matching_nvs <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L572" class="blob-num js-line-number" data-line-number="572"></td>
        <td id="LC572" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> v1_test, v2_test <span class="pl-k">in</span> [[v1, v2], [v2, v3], [v3, <span class="pl-k">-</span>v1], [<span class="pl-k">-</span>v1, <span class="pl-k">-</span>v2], [<span class="pl-k">-</span>v2, <span class="pl-k">-</span>v3], [<span class="pl-k">-</span>v3, v1]]:</td>
      </tr>
      <tr>
        <td id="L573" class="blob-num js-line-number" data-line-number="573"></td>
        <td id="LC573" class="blob-code blob-code-inner js-file-line">                v_matrix <span class="pl-k">=</span> np.array([v1_test, v2_test]).transpose()</td>
      </tr>
      <tr>
        <td id="L574" class="blob-num js-line-number" data-line-number="574"></td>
        <td id="LC574" class="blob-code blob-code-inner js-file-line">                pixels_fit_nvs <span class="pl-k">=</span> np.round(np.linalg.pinv(v_matrix).dot((pixels_fit_px <span class="pl-k">-</span> <span class="pl-c1">self</span>.center_fitted).transpose()))</td>
      </tr>
      <tr>
        <td id="L575" class="blob-num js-line-number" data-line-number="575"></td>
        <td id="LC575" class="blob-code blob-code-inner js-file-line">                pixels_fit_nvs_set <span class="pl-k">=</span> <span class="pl-c1">set</span>(<span class="pl-c1">map</span>(<span class="pl-c1">tuple</span>, pixels_fit_nvs.transpose().astype(<span class="pl-c1">int</span>)))</td>
      </tr>
      <tr>
        <td id="L576" class="blob-num js-line-number" data-line-number="576"></td>
        <td id="LC576" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">for</span> nv1 <span class="pl-k">in</span> <span class="pl-c1">range</span>(<span class="pl-k">-</span><span class="pl-c1">10</span>, <span class="pl-c1">10</span>):</td>
      </tr>
      <tr>
        <td id="L577" class="blob-num js-line-number" data-line-number="577"></td>
        <td id="LC577" class="blob-code blob-code-inner js-file-line">                    <span class="pl-k">for</span> nv2 <span class="pl-k">in</span> <span class="pl-c1">range</span>(<span class="pl-k">-</span><span class="pl-c1">10</span>, <span class="pl-c1">10</span>):</td>
      </tr>
      <tr>
        <td id="L578" class="blob-num js-line-number" data-line-number="578"></td>
        <td id="LC578" class="blob-code blob-code-inner js-file-line">                        pixels_nvs <span class="pl-k">=</span> <span class="pl-c1">map</span>(<span class="pl-c1">tuple</span>, <span class="pl-c1">self</span>.pixels_nvs.transpose() <span class="pl-k">+</span> np.array([nv1, nv2]))</td>
      </tr>
      <tr>
        <td id="L579" class="blob-num js-line-number" data-line-number="579"></td>
        <td id="LC579" class="blob-code blob-code-inner js-file-line">                        matching_nvs <span class="pl-k">=</span> <span class="pl-c1">len</span>(pixels_fit_nvs_set.intersection(pixels_nvs))</td>
      </tr>
      <tr>
        <td id="L580" class="blob-num js-line-number" data-line-number="580"></td>
        <td id="LC580" class="blob-code blob-code-inner js-file-line">                        <span class="pl-k">if</span> matching_nvs <span class="pl-k">&gt;</span> best_matching_nvs:</td>
      </tr>
      <tr>
        <td id="L581" class="blob-num js-line-number" data-line-number="581"></td>
        <td id="LC581" class="blob-code blob-code-inner js-file-line">                            best_matching_nvs <span class="pl-k">=</span> matching_nvs</td>
      </tr>
      <tr>
        <td id="L582" class="blob-num js-line-number" data-line-number="582"></td>
        <td id="LC582" class="blob-code blob-code-inner js-file-line">                            best_match_nv1, best_match_nv2 <span class="pl-k">=</span> nv1, nv2</td>
      </tr>
      <tr>
        <td id="L583" class="blob-num js-line-number" data-line-number="583"></td>
        <td id="LC583" class="blob-code blob-code-inner js-file-line">                            best_match_v1, best_match_v2 <span class="pl-k">=</span>  v1_test, v2_test</td>
      </tr>
      <tr>
        <td id="L584" class="blob-num js-line-number" data-line-number="584"></td>
        <td id="LC584" class="blob-code blob-code-inner js-file-line">            v1, v2 <span class="pl-k">=</span> best_match_v1, best_match_v2</td>
      </tr>
      <tr>
        <td id="L585" class="blob-num js-line-number" data-line-number="585"></td>
        <td id="LC585" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>best base: v1=<span class="pl-pds">&#39;</span></span>, v1, <span class="pl-s"><span class="pl-pds">&#39;</span>, v2=<span class="pl-pds">&#39;</span></span>, v2)</td>
      </tr>
      <tr>
        <td id="L586" class="blob-num js-line-number" data-line-number="586"></td>
        <td id="LC586" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>best match: nv1 =<span class="pl-pds">&#39;</span></span>, best_match_nv1, <span class="pl-s"><span class="pl-pds">&#39;</span>, nv2 =<span class="pl-pds">&#39;</span></span>, best_match_nv2)</td>
      </tr>
      <tr>
        <td id="L587" class="blob-num js-line-number" data-line-number="587"></td>
        <td id="LC587" class="blob-code blob-code-inner js-file-line">            v_matrix <span class="pl-k">=</span> np.array([v1, v2]).transpose()</td>
      </tr>
      <tr>
        <td id="L588" class="blob-num js-line-number" data-line-number="588"></td>
        <td id="LC588" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.center_fitted <span class="pl-k">+=</span> best_match_nv1 <span class="pl-k">*</span> v1 <span class="pl-k">+</span> best_match_nv2 <span class="pl-k">*</span> v2</td>
      </tr>
      <tr>
        <td id="L589" class="blob-num js-line-number" data-line-number="589"></td>
        <td id="LC589" class="blob-code blob-code-inner js-file-line">            pixels_fit_nvs <span class="pl-k">=</span> np.round(np.linalg.pinv(v_matrix).dot((pixels_fit_px <span class="pl-k">-</span> <span class="pl-c1">self</span>.center_fitted).transpose()))</td>
      </tr>
      <tr>
        <td id="L590" class="blob-num js-line-number" data-line-number="590"></td>
        <td id="LC590" class="blob-code blob-code-inner js-file-line">            pixels_fit_nvs_set <span class="pl-k">=</span> <span class="pl-c1">set</span>(<span class="pl-c1">map</span>(<span class="pl-c1">tuple</span>, pixels_fit_nvs.transpose().astype(<span class="pl-c1">int</span>)))</td>
      </tr>
      <tr>
        <td id="L591" class="blob-num js-line-number" data-line-number="591"></td>
        <td id="LC591" class="blob-code blob-code-inner js-file-line">            nvs_fit_matching_set <span class="pl-k">=</span> pixels_fit_nvs_set.intersection(<span class="pl-c1">map</span>(<span class="pl-c1">tuple</span>, <span class="pl-c1">self</span>.pixels_nvs.transpose()))</td>
      </tr>
      <tr>
        <td id="L592" class="blob-num js-line-number" data-line-number="592"></td>
        <td id="LC592" class="blob-code blob-code-inner js-file-line">            is_fit_matching <span class="pl-k">=</span> np.array([x <span class="pl-k">in</span> nvs_fit_matching_set <span class="pl-k">for</span> x <span class="pl-k">in</span> <span class="pl-c1">map</span>(<span class="pl-c1">tuple</span>, pixels_fit_nvs.transpose())])</td>
      </tr>
      <tr>
        <td id="L593" class="blob-num js-line-number" data-line-number="593"></td>
        <td id="LC593" class="blob-code blob-code-inner js-file-line">            fit_not_matching <span class="pl-k">=</span> is_fit_matching <span class="pl-k">==</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L594" class="blob-num js-line-number" data-line-number="594"></td>
        <td id="LC594" class="blob-code blob-code-inner js-file-line">            is_pixel_fitted <span class="pl-k">=</span> np.array([x <span class="pl-k">in</span> nvs_fit_matching_set <span class="pl-k">for</span> x <span class="pl-k">in</span> <span class="pl-c1">map</span>(<span class="pl-c1">tuple</span>, <span class="pl-c1">self</span>.pixels_nvs.transpose())])</td>
      </tr>
      <tr>
        <td id="L595" class="blob-num js-line-number" data-line-number="595"></td>
        <td id="LC595" class="blob-code blob-code-inner js-file-line">            pixels_not_fitted <span class="pl-k">=</span> is_pixel_fitted <span class="pl-k">==</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L596" class="blob-num js-line-number" data-line-number="596"></td>
        <td id="LC596" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(np.sum(is_fit_matching), <span class="pl-s"><span class="pl-pds">&#39;</span>fits in best match,<span class="pl-pds">&#39;</span></span>, np.sum(fit_not_matching), <span class="pl-s"><span class="pl-pds">&#39;</span>outside<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L597" class="blob-num js-line-number" data-line-number="597"></td>
        <td id="LC597" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>global fit of lattice base vectors, center was <span class="pl-pds">&#39;</span></span>, <span class="pl-c1">self</span>.center_fitted)</td>
      </tr>
      <tr>
        <td id="L598" class="blob-num js-line-number" data-line-number="598"></td>
        <td id="LC598" class="blob-code blob-code-inner js-file-line">            n_matching <span class="pl-k">=</span> np.sum(is_fit_matching)</td>
      </tr>
      <tr>
        <td id="L599" class="blob-num js-line-number" data-line-number="599"></td>
        <td id="LC599" class="blob-code blob-code-inner js-file-line">            nvs <span class="pl-k">=</span> np.vstack((pixels_fit_nvs[:, is_fit_matching], np.ones((<span class="pl-c1">1</span>, n_matching))))</td>
      </tr>
      <tr>
        <td id="L600" class="blob-num js-line-number" data-line-number="600"></td>
        <td id="LC600" class="blob-code blob-code-inner js-file-line">            pxs <span class="pl-k">=</span> pixels_fit_px[is_fit_matching, :].transpose()</td>
      </tr>
      <tr>
        <td id="L601" class="blob-num js-line-number" data-line-number="601"></td>
        <td id="LC601" class="blob-code blob-code-inner js-file-line">            precise_vs <span class="pl-k">=</span> pxs.dot(np.linalg.pinv(nvs))</td>
      </tr>
      <tr>
        <td id="L602" class="blob-num js-line-number" data-line-number="602"></td>
        <td id="LC602" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.v1_lattice <span class="pl-k">=</span> precise_vs[:, <span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L603" class="blob-num js-line-number" data-line-number="603"></td>
        <td id="LC603" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.v2_lattice <span class="pl-k">=</span> precise_vs[:, <span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L604" class="blob-num js-line-number" data-line-number="604"></td>
        <td id="LC604" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.center_fitted <span class="pl-k">=</span> precise_vs[:, <span class="pl-c1">2</span>]</td>
      </tr>
      <tr>
        <td id="L605" class="blob-num js-line-number" data-line-number="605"></td>
        <td id="LC605" class="blob-code blob-code-inner js-file-line">            pixels_geom_px <span class="pl-k">=</span> precise_vs.dot(np.vstack((<span class="pl-c1">self</span>.pixels_nvs,  np.ones((<span class="pl-c1">1</span>, <span class="pl-c1">self</span>.pixels_nvs.shape[<span class="pl-c1">1</span>])))))</td>
      </tr>
      <tr>
        <td id="L606" class="blob-num js-line-number" data-line-number="606"></td>
        <td id="LC606" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>v1=<span class="pl-pds">&quot;</span></span>, <span class="pl-c1">self</span>.v1_lattice, <span class="pl-s"><span class="pl-pds">&quot;</span>|v1|=<span class="pl-pds">&quot;</span></span>, np.abs(<span class="pl-c1">self</span>.v1_lattice[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.v1_lattice[<span class="pl-c1">1</span>]))</td>
      </tr>
      <tr>
        <td id="L607" class="blob-num js-line-number" data-line-number="607"></td>
        <td id="LC607" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>v2=<span class="pl-pds">&quot;</span></span>, <span class="pl-c1">self</span>.v2_lattice, <span class="pl-s"><span class="pl-pds">&quot;</span>|v2|=<span class="pl-pds">&quot;</span></span>, np.abs(<span class="pl-c1">self</span>.v2_lattice[<span class="pl-c1">0</span>] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.v2_lattice[<span class="pl-c1">1</span>]))</td>
      </tr>
      <tr>
        <td id="L608" class="blob-num js-line-number" data-line-number="608"></td>
        <td id="LC608" class="blob-code blob-code-inner js-file-line">            center_image <span class="pl-k">=</span> (np.array(<span class="pl-c1">self</span>.image_cones.shape[::<span class="pl-k">-</span><span class="pl-c1">1</span>]) <span class="pl-k">-</span><span class="pl-c1">1</span>) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L609" class="blob-num js-line-number" data-line-number="609"></td>
        <td id="LC609" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>center=<span class="pl-pds">&quot;</span></span>, <span class="pl-c1">self</span>.center_fitted, <span class="pl-s"><span class="pl-pds">&#39;</span>,<span class="pl-pds">&#39;</span></span>, <span class="pl-c1">self</span>.center_fitted <span class="pl-k">-</span> center_image, <span class="pl-s"><span class="pl-pds">&#39;</span>from center<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L610" class="blob-num js-line-number" data-line-number="610"></td>
        <td id="LC610" class="blob-code blob-code-inner js-file-line">            fig <span class="pl-k">=</span> plt.figure()</td>
      </tr>
      <tr>
        <td id="L611" class="blob-num js-line-number" data-line-number="611"></td>
        <td id="LC611" class="blob-code blob-code-inner js-file-line">            ax <span class="pl-k">=</span> plt.gca()</td>
      </tr>
      <tr>
        <td id="L612" class="blob-num js-line-number" data-line-number="612"></td>
        <td id="LC612" class="blob-code blob-code-inner js-file-line">            plt.imshow(cone_presence, <span class="pl-v">cmap</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>gray<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L613" class="blob-num js-line-number" data-line-number="613"></td>
        <td id="LC613" class="blob-code blob-code-inner js-file-line">            plt.autoscale(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L614" class="blob-num js-line-number" data-line-number="614"></td>
        <td id="LC614" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> center_pixel <span class="pl-k">in</span> pixels_geom_px[:, is_pixel_fitted].transpose():</td>
      </tr>
      <tr>
        <td id="L615" class="blob-num js-line-number" data-line-number="615"></td>
        <td id="LC615" class="blob-code blob-code-inner js-file-line">                circle <span class="pl-k">=</span> Circle((center_pixel[<span class="pl-c1">0</span>], center_pixel[<span class="pl-c1">1</span>]), <span class="pl-v">radius</span><span class="pl-k">=</span>radius_mask <span class="pl-k">+</span> <span class="pl-c1">10</span>, <span class="pl-v">fill</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-v">color</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>g<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L616" class="blob-num js-line-number" data-line-number="616"></td>
        <td id="LC616" class="blob-code blob-code-inner js-file-line">                ax.add_artist(circle)</td>
      </tr>
      <tr>
        <td id="L617" class="blob-num js-line-number" data-line-number="617"></td>
        <td id="LC617" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> center_pixel <span class="pl-k">in</span> pixels_geom_px[:, pixels_not_fitted].transpose():</td>
      </tr>
      <tr>
        <td id="L618" class="blob-num js-line-number" data-line-number="618"></td>
        <td id="LC618" class="blob-code blob-code-inner js-file-line">                circle <span class="pl-k">=</span> Circle((center_pixel[<span class="pl-c1">0</span>], center_pixel[<span class="pl-c1">1</span>]), <span class="pl-v">radius</span><span class="pl-k">=</span>radius_mask <span class="pl-k">+</span> <span class="pl-c1">10</span>, <span class="pl-v">fill</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-v">color</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>r<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L619" class="blob-num js-line-number" data-line-number="619"></td>
        <td id="LC619" class="blob-code blob-code-inner js-file-line">                ax.add_artist(circle)</td>
      </tr>
      <tr>
        <td id="L620" class="blob-num js-line-number" data-line-number="620"></td>
        <td id="LC620" class="blob-code blob-code-inner js-file-line">            plt.errorbar(pixels_fit_px[is_fit_matching, <span class="pl-c1">0</span>], pixels_fit_px[is_fit_matching, <span class="pl-c1">1</span>],</td>
      </tr>
      <tr>
        <td id="L621" class="blob-num js-line-number" data-line-number="621"></td>
        <td id="LC621" class="blob-code blob-code-inner js-file-line">                         <span class="pl-v">xerr</span><span class="pl-k">=</span>pixels_fit_sigma[is_fit_matching, <span class="pl-c1">0</span>] <span class="pl-k">*</span> <span class="pl-c1">3</span>, <span class="pl-v">yerr</span><span class="pl-k">=</span>pixels_fit_sigma[is_fit_matching, <span class="pl-c1">1</span>] <span class="pl-k">*</span> <span class="pl-c1">3</span>,</td>
      </tr>
      <tr>
        <td id="L622" class="blob-num js-line-number" data-line-number="622"></td>
        <td id="LC622" class="blob-code blob-code-inner js-file-line">                         <span class="pl-v">fmt</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>b<span class="pl-pds">&#39;</span></span>, <span class="pl-v">linestyle</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>none<span class="pl-pds">&#39;</span></span>, <span class="pl-v">elinewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L623" class="blob-num js-line-number" data-line-number="623"></td>
        <td id="LC623" class="blob-code blob-code-inner js-file-line">            plt.errorbar(pixels_fit_px[fit_not_matching, <span class="pl-c1">0</span>], pixels_fit_px[fit_not_matching, <span class="pl-c1">1</span>],</td>
      </tr>
      <tr>
        <td id="L624" class="blob-num js-line-number" data-line-number="624"></td>
        <td id="LC624" class="blob-code blob-code-inner js-file-line">                         <span class="pl-v">xerr</span><span class="pl-k">=</span>pixels_fit_sigma[fit_not_matching, <span class="pl-c1">0</span>] <span class="pl-k">*</span> <span class="pl-c1">3</span>, <span class="pl-v">yerr</span><span class="pl-k">=</span>pixels_fit_sigma[fit_not_matching, <span class="pl-c1">1</span>] <span class="pl-k">*</span> <span class="pl-c1">3</span>,</td>
      </tr>
      <tr>
        <td id="L625" class="blob-num js-line-number" data-line-number="625"></td>
        <td id="LC625" class="blob-code blob-code-inner js-file-line">                         <span class="pl-v">fmt</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>y<span class="pl-pds">&#39;</span></span>, <span class="pl-v">linestyle</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>none<span class="pl-pds">&#39;</span></span>, <span class="pl-v">elinewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L626" class="blob-num js-line-number" data-line-number="626"></td>
        <td id="LC626" class="blob-code blob-code-inner js-file-line">            plt.grid(<span class="pl-c1">None</span>)</td>
      </tr>
      <tr>
        <td id="L627" class="blob-num js-line-number" data-line-number="627"></td>
        <td id="LC627" class="blob-code blob-code-inner js-file-line">            plt.axis(<span class="pl-s"><span class="pl-pds">&#39;</span>off<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L628" class="blob-num js-line-number" data-line-number="628"></td>
        <td id="LC628" class="blob-code blob-code-inner js-file-line">            ax.get_xaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L629" class="blob-num js-line-number" data-line-number="629"></td>
        <td id="LC629" class="blob-code blob-code-inner js-file-line">            ax.get_yaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L630" class="blob-num js-line-number" data-line-number="630"></td>
        <td id="LC630" class="blob-code blob-code-inner js-file-line">            pixels_px_prediction <span class="pl-k">=</span> precise_vs.dot(nvs)</td>
      </tr>
      <tr>
        <td id="L631" class="blob-num js-line-number" data-line-number="631"></td>
        <td id="LC631" class="blob-code blob-code-inner js-file-line">            residuals <span class="pl-k">=</span> pxs <span class="pl-k">-</span> pixels_px_prediction</td>
      </tr>
      <tr>
        <td id="L632" class="blob-num js-line-number" data-line-number="632"></td>
        <td id="LC632" class="blob-code blob-code-inner js-file-line">            good_residuals <span class="pl-k">=</span> np.abs(residuals[<span class="pl-c1">0</span>, :] <span class="pl-k">+</span> <span class="pl-c1">1<span class="pl-k">j</span></span> <span class="pl-k">*</span> residuals[<span class="pl-c1">1</span>, :]) <span class="pl-k">&lt;</span> <span class="pl-c1">2</span>.</td>
      </tr>
      <tr>
        <td id="L633" class="blob-num js-line-number" data-line-number="633"></td>
        <td id="LC633" class="blob-code blob-code-inner js-file-line">            bad_residuals <span class="pl-k">=</span> good_residuals <span class="pl-k">==</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L634" class="blob-num js-line-number" data-line-number="634"></td>
        <td id="LC634" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(np.sum(good_residuals), <span class="pl-s"><span class="pl-pds">&quot;</span>residuals below threshold,<span class="pl-pds">&quot;</span></span>, np.sum(bad_residuals), <span class="pl-s"><span class="pl-pds">&quot;</span>above.<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L635" class="blob-num js-line-number" data-line-number="635"></td>
        <td id="LC635" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>mean(residuals)=<span class="pl-pds">&#39;</span></span>, np.mean(residuals[:, good_residuals], <span class="pl-v">axis</span><span class="pl-k">=</span><span class="pl-c1">1</span>),</td>
      </tr>
      <tr>
        <td id="L636" class="blob-num js-line-number" data-line-number="636"></td>
        <td id="LC636" class="blob-code blob-code-inner js-file-line">                  <span class="pl-s"><span class="pl-pds">&#39;</span>std(residual)=<span class="pl-pds">&#39;</span></span>, np.std(residuals[:, good_residuals], <span class="pl-v">axis</span><span class="pl-k">=</span><span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L637" class="blob-num js-line-number" data-line-number="637"></td>
        <td id="LC637" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L638" class="blob-num js-line-number" data-line-number="638"></td>
        <td id="LC638" class="blob-code blob-code-inner js-file-line">                plt.show()</td>
      </tr>
      <tr>
        <td id="L639" class="blob-num js-line-number" data-line-number="639"></td>
        <td id="LC639" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L640" class="blob-num js-line-number" data-line-number="640"></td>
        <td id="LC640" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> <span class="pl-c1">self</span>.filename <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L641" class="blob-num js-line-number" data-line-number="641"></td>
        <td id="LC641" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> <span class="pl-c1">self</span>.filename.replace(<span class="pl-s"><span class="pl-pds">&#39;</span>.fits<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>-cones-presence-filtered.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L642" class="blob-num js-line-number" data-line-number="642"></td>
        <td id="LC642" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L643" class="blob-num js-line-number" data-line-number="643"></td>
        <td id="LC643" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>cones-presence-filtered.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L644" class="blob-num js-line-number" data-line-number="644"></td>
        <td id="LC644" class="blob-code blob-code-inner js-file-line">                plt.savefig(output_filename, <span class="pl-v">bbox_inches</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>tight<span class="pl-pds">&#39;</span></span>, <span class="pl-v">pad_inches</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L645" class="blob-num js-line-number" data-line-number="645"></td>
        <td id="LC645" class="blob-code blob-code-inner js-file-line">                plt.close(fig)</td>
      </tr>
      <tr>
        <td id="L646" class="blob-num js-line-number" data-line-number="646"></td>
        <td id="LC646" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">print</span>(output_filename, <span class="pl-s"><span class="pl-pds">&#39;</span>saved.<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L647" class="blob-num js-line-number" data-line-number="647"></td>
        <td id="LC647" class="blob-code blob-code-inner js-file-line">            fig <span class="pl-k">=</span> plt.figure()</td>
      </tr>
      <tr>
        <td id="L648" class="blob-num js-line-number" data-line-number="648"></td>
        <td id="LC648" class="blob-code blob-code-inner js-file-line">            ax <span class="pl-k">=</span> plt.gca()</td>
      </tr>
      <tr>
        <td id="L649" class="blob-num js-line-number" data-line-number="649"></td>
        <td id="LC649" class="blob-code blob-code-inner js-file-line">            plt.imshow(<span class="pl-c1">self</span>.image_cones, <span class="pl-v">cmap</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>gray<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L650" class="blob-num js-line-number" data-line-number="650"></td>
        <td id="LC650" class="blob-code blob-code-inner js-file-line">            plt.autoscale(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L651" class="blob-num js-line-number" data-line-number="651"></td>
        <td id="LC651" class="blob-code blob-code-inner js-file-line">            ploted_residuals <span class="pl-k">=</span> residuals[:, good_residuals].transpose()</td>
      </tr>
      <tr>
        <td id="L652" class="blob-num js-line-number" data-line-number="652"></td>
        <td id="LC652" class="blob-code blob-code-inner js-file-line">            ploted_prediction <span class="pl-k">=</span> pixels_px_prediction[:, good_residuals].transpose()</td>
      </tr>
      <tr>
        <td id="L653" class="blob-num js-line-number" data-line-number="653"></td>
        <td id="LC653" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> residual, center_pixel <span class="pl-k">in</span> <span class="pl-c1">zip</span>(ploted_residuals, ploted_prediction):</td>
      </tr>
      <tr>
        <td id="L654" class="blob-num js-line-number" data-line-number="654"></td>
        <td id="LC654" class="blob-code blob-code-inner js-file-line">                arrow <span class="pl-k">=</span> Arrow(center_pixel[<span class="pl-c1">0</span>], center_pixel[<span class="pl-c1">1</span>], residual[<span class="pl-c1">0</span>]<span class="pl-k">*</span><span class="pl-c1">10</span>, residual[<span class="pl-c1">1</span>]<span class="pl-k">*</span><span class="pl-c1">10</span>,</td>
      </tr>
      <tr>
        <td id="L655" class="blob-num js-line-number" data-line-number="655"></td>
        <td id="LC655" class="blob-code blob-code-inner js-file-line">                              <span class="pl-v">color</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>b<span class="pl-pds">&#39;</span></span>, <span class="pl-v">width</span><span class="pl-k">=</span><span class="pl-c1">15</span>., <span class="pl-v">linewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L656" class="blob-num js-line-number" data-line-number="656"></td>
        <td id="LC656" class="blob-code blob-code-inner js-file-line">                ax.add_artist(arrow)</td>
      </tr>
      <tr>
        <td id="L657" class="blob-num js-line-number" data-line-number="657"></td>
        <td id="LC657" class="blob-code blob-code-inner js-file-line">            ploted_residuals <span class="pl-k">=</span> residuals[:, bad_residuals].transpose()</td>
      </tr>
      <tr>
        <td id="L658" class="blob-num js-line-number" data-line-number="658"></td>
        <td id="LC658" class="blob-code blob-code-inner js-file-line">            ploted_prediction <span class="pl-k">=</span> pixels_px_prediction[:, bad_residuals].transpose()</td>
      </tr>
      <tr>
        <td id="L659" class="blob-num js-line-number" data-line-number="659"></td>
        <td id="LC659" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> residual, center_pixel <span class="pl-k">in</span> <span class="pl-c1">zip</span>(ploted_residuals, ploted_prediction):</td>
      </tr>
      <tr>
        <td id="L660" class="blob-num js-line-number" data-line-number="660"></td>
        <td id="LC660" class="blob-code blob-code-inner js-file-line">                arrow <span class="pl-k">=</span> Arrow(center_pixel[<span class="pl-c1">0</span>], center_pixel[<span class="pl-c1">1</span>], residual[<span class="pl-c1">0</span>]<span class="pl-k">*</span><span class="pl-c1">10</span>, residual[<span class="pl-c1">1</span>]<span class="pl-k">*</span><span class="pl-c1">10</span>,</td>
      </tr>
      <tr>
        <td id="L661" class="blob-num js-line-number" data-line-number="661"></td>
        <td id="LC661" class="blob-code blob-code-inner js-file-line">                              <span class="pl-v">color</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>r<span class="pl-pds">&#39;</span></span>, <span class="pl-v">width</span><span class="pl-k">=</span><span class="pl-c1">15</span>., <span class="pl-v">linewidth</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L662" class="blob-num js-line-number" data-line-number="662"></td>
        <td id="LC662" class="blob-code blob-code-inner js-file-line">                ax.add_artist(arrow)</td>
      </tr>
      <tr>
        <td id="L663" class="blob-num js-line-number" data-line-number="663"></td>
        <td id="LC663" class="blob-code blob-code-inner js-file-line">            plt.grid(<span class="pl-c1">None</span>)</td>
      </tr>
      <tr>
        <td id="L664" class="blob-num js-line-number" data-line-number="664"></td>
        <td id="LC664" class="blob-code blob-code-inner js-file-line">            plt.axis(<span class="pl-s"><span class="pl-pds">&#39;</span>off<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L665" class="blob-num js-line-number" data-line-number="665"></td>
        <td id="LC665" class="blob-code blob-code-inner js-file-line">            ax.get_xaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L666" class="blob-num js-line-number" data-line-number="666"></td>
        <td id="LC666" class="blob-code blob-code-inner js-file-line">            ax.get_yaxis().set_visible(<span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L667" class="blob-num js-line-number" data-line-number="667"></td>
        <td id="LC667" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> output_dir <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L668" class="blob-num js-line-number" data-line-number="668"></td>
        <td id="LC668" class="blob-code blob-code-inner js-file-line">                plt.show()</td>
      </tr>
      <tr>
        <td id="L669" class="blob-num js-line-number" data-line-number="669"></td>
        <td id="LC669" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L670" class="blob-num js-line-number" data-line-number="670"></td>
        <td id="LC670" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> <span class="pl-c1">self</span>.filename <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L671" class="blob-num js-line-number" data-line-number="671"></td>
        <td id="LC671" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> <span class="pl-c1">self</span>.filename.replace(<span class="pl-s"><span class="pl-pds">&#39;</span>.fits<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>-cones-fit-residuals.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L672" class="blob-num js-line-number" data-line-number="672"></td>
        <td id="LC672" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L673" class="blob-num js-line-number" data-line-number="673"></td>
        <td id="LC673" class="blob-code blob-code-inner js-file-line">                    output_filename <span class="pl-k">=</span> os.path.join(output_dir, <span class="pl-s"><span class="pl-pds">&#39;</span>cones-fit-residuals.png<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L674" class="blob-num js-line-number" data-line-number="674"></td>
        <td id="LC674" class="blob-code blob-code-inner js-file-line">                plt.savefig(output_filename, <span class="pl-v">bbox_inches</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>tight<span class="pl-pds">&#39;</span></span>, <span class="pl-v">pad_inches</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L675" class="blob-num js-line-number" data-line-number="675"></td>
        <td id="LC675" class="blob-code blob-code-inner js-file-line">                plt.close(fig)</td>
      </tr>
      <tr>
        <td id="L676" class="blob-num js-line-number" data-line-number="676"></td>
        <td id="LC676" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">print</span>(output_filename, <span class="pl-s"><span class="pl-pds">&#39;</span>saved.<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L677" class="blob-num js-line-number" data-line-number="677"></td>
        <td id="LC677" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
</table>

  <div class="BlobToolbar position-absolute js-file-line-actions dropdown js-menu-container js-select-menu d-none" aria-hidden="true">
    <button class="btn-octicon ml-0 px-2 p-0 bg-white border border-gray-dark rounded-1 dropdown-toggle js-menu-target" id="js-file-line-action-button" type="button" aria-expanded="false" aria-haspopup="true" aria-label="Inline file action toolbar" aria-controls="inline-file-actions">
      <svg aria-hidden="true" class="octicon octicon-kebab-horizontal" height="16" version="1.1" viewBox="0 0 13 16" width="13"><path fill-rule="evenodd" d="M1.5 9a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3zm5 0a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3zm5 0a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3z"/></svg>
    </button>
    <div class="dropdown-menu-content js-menu-content" id="inline-file-actions">
      <ul class="BlobToolbar-dropdown dropdown-menu dropdown-menu-se mt-2">
        <li><a class="js-zeroclipboard dropdown-item" style="cursor:pointer;" id="js-copy-lines" data-original-text="Copy lines">Copy lines</a></li>
        <li><a class="js-zeroclipboard dropdown-item" id= "js-copy-permalink" style="cursor:pointer;" data-original-text="Copy permalink">Copy permalink</a></li>
        <li><a href="/calispac/digicampipe/blame/57d647885a442b0c1f88566b8c2a8869e1eb5e2d/digicampipe/image/cones_image.py" class="dropdown-item js-update-url-with-hash" id="js-view-git-blame">View git blame</a></li>
          <li><a href="/calispac/digicampipe/issues/new" class="dropdown-item" id="js-new-issue">Open new issue</a></li>
      </ul>
    </div>
  </div>

  </div>

  </div>

  <button type="button" data-facebox="#jump-to-line" data-facebox-class="linejump" data-hotkey="l" class="d-none">Jump to Line</button>
  <div id="jump-to-line" style="display:none">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="" class="js-jump-to-line-form" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /></div>
      <input class="form-control linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
      <button type="submit" class="btn">Go</button>
</form>  </div>


  </div>
  <div class="modal-backdrop js-touch-events"></div>
</div>

    </div>
  </div>

  </div>

      
<div class="footer container-lg px-3" role="contentinfo">
  <div class="position-relative d-flex flex-justify-between py-6 mt-6 f6 text-gray border-top border-gray-light ">
    <ul class="list-style-none d-flex flex-wrap ">
      <li class="mr-3">&copy; 2017 <span title="0.28008s from unicorn-1183575238-nk723">GitHub</span>, Inc.</li>
        <li class="mr-3"><a href="https://github.com/site/terms" data-ga-click="Footer, go to terms, text:terms">Terms</a></li>
        <li class="mr-3"><a href="https://github.com/site/privacy" data-ga-click="Footer, go to privacy, text:privacy">Privacy</a></li>
        <li class="mr-3"><a href="https://github.com/security" data-ga-click="Footer, go to security, text:security">Security</a></li>
        <li class="mr-3"><a href="https://status.github.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
        <li><a href="https://help.github.com" data-ga-click="Footer, go to help, text:help">Help</a></li>
    </ul>

    <a href="https://github.com" aria-label="Homepage" class="footer-octicon" title="GitHub">
      <svg aria-hidden="true" class="octicon octicon-mark-github" height="24" version="1.1" viewBox="0 0 16 16" width="24"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
</a>
    <ul class="list-style-none d-flex flex-wrap ">
        <li class="mr-3"><a href="https://github.com/contact" data-ga-click="Footer, go to contact, text:contact">Contact GitHub</a></li>
      <li class="mr-3"><a href="https://developer.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li class="mr-3"><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
      <li class="mr-3"><a href="https://shop.github.com" data-ga-click="Footer, go to shop, text:shop">Shop</a></li>
        <li class="mr-3"><a href="https://github.com/blog" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a href="https://github.com/about" data-ga-click="Footer, go to about, text:about">About</a></li>

    </ul>
  </div>
</div>



  <div id="ajax-error-message" class="ajax-error-message flash flash-error">
    <svg aria-hidden="true" class="octicon octicon-alert" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.865 1.52c-.18-.31-.51-.5-.87-.5s-.69.19-.87.5L.275 13.5c-.18.31-.18.69 0 1 .19.31.52.5.87.5h13.7c.36 0 .69-.19.86-.5.17-.31.18-.69.01-1L8.865 1.52zM8.995 13h-2v-2h2v2zm0-3h-2V6h2v4z"/></svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg aria-hidden="true" class="octicon octicon-x" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
    </button>
    You can't perform that action at this time.
  </div>


    <script crossorigin="anonymous" src="https://assets-cdn.github.com/assets/compat-e42a8bf9c380758734e39851db04de7cbeeb2f3860efbd481c96ac12c25a6ecb.js"></script>
    <script crossorigin="anonymous" src="https://assets-cdn.github.com/assets/frameworks-494003167c53469f14f39a4d07f13f27dc10c67f09b306dc4460b6030558210b.js"></script>
    
    <script async="async" crossorigin="anonymous" src="https://assets-cdn.github.com/assets/github-0637c99f4793f245d18479f8483e1e29834e15e5b71456dd40b572364d5f82cf.js"></script>
    
    
    
    
  <div class="js-stale-session-flash stale-session-flash flash flash-warn flash-banner d-none">
    <svg aria-hidden="true" class="octicon octicon-alert" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.865 1.52c-.18-.31-.51-.5-.87-.5s-.69.19-.87.5L.275 13.5c-.18.31-.18.69 0 1 .19.31.52.5.87.5h13.7c.36 0 .69-.19.86-.5.17-.31.18-.69.01-1L8.865 1.52zM8.995 13h-2v-2h2v2zm0-3h-2V6h2v4z"/></svg>
    <span class="signed-in-tab-flash">You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
    <span class="signed-out-tab-flash">You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
  </div>
  <div class="facebox" id="facebox" style="display:none;">
  <div class="facebox-popup">
    <div class="facebox-content" role="dialog" aria-labelledby="facebox-header" aria-describedby="facebox-description">
    </div>
    <button type="button" class="facebox-close js-facebox-close" aria-label="Close modal">
      <svg aria-hidden="true" class="octicon octicon-x" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
    </button>
  </div>
</div>


  </body>
</html>

