TODO: INTRODUCTION
TODO: HOW IT WORKS
TODO: CODE EXAMPLES


TODO
  - Review setuptools/pip/installation stuff
  - Give an overview of how the boxeye works and how it is structured.
  - Better test coverage (currently minimal)
  - Improve documentation, cleanup related files.
  - Dependency injection for `capture`, `click`, and `drag`.
    + ie `boxeye.inject(dependency)` -- is this possible?
  - rewrite (v1) without using classes ?
   

Boxeye is currently in alpha (v0.0.3).  The interface has solidified
pretty much, all Pattern subclasses provide a `locate` and a `isvisible`
function.  `locate` returns a list of matches.  `isvisible` runs `locate`
and returns `True` if there are matches.


Compatible with Android devices.  
