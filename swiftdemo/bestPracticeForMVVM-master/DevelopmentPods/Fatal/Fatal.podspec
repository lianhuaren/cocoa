#
# Be sure to run `pod lib lint Fatal.podspec' to ensure this is a
# valid spec before submitting.
#
# Any lines starting with a # are optional, but their use is encouraged
# To learn more about a Podspec see https://guides.cocoapods.org/syntax/podspec.html
#

Pod::Spec.new do |s|
  
  s.name             = 'Fatal'
  s.version          = '1.0.0'
  s.summary          = 'Application error definition module.'
  
  s.homepage         = 'Coming soon...'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Archer' => 'code4archer@163.com' }
  s.source           = { :git => 'Coming soon...', :tag => s.version.to_s }
  
  s.ios.deployment_target = '8.0'
  
  s.source_files  = "Fatal/Classes/*.swift"
  
end
