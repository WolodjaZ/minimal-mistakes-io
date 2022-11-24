---
title:  "Blogs"
layout: archive
permalink: /Blogs/
author_profile: true
comments: true
---

Most of my blogs are technical blogs written mainly for my own reference. I'd be happy if any of you find them useful too.

{% include base_path %}

<table style="border: 0; border-collapse: separate; border-spacing: 0 25px;">
  {% assign years = site.posts | group_by: "year" | sort: "name" | reverse %}
  {% for y in years %}
    {% assign sorted = y.items | sort: "venue" | reverse %}
    {% for post in sorted %}
      {% include archive-single.html %}
    {% endfor %}
  {% endfor %}
</table>