package bootstrap

import (
	"mcp/config/interf"
	"path"
)

// Configurator 是一个接口，定义了创建配置的方法
func InitConfig(FileName string) interf.ConfigInterface {

	//读取后缀
	FileType := path.Ext(FileName)
	cfg := interf.CreateConfigFactory(FileName, FileType[1:])
	return cfg
}
