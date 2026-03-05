package interf

import (
	"github.com/spf13/viper"
	"mcp/global"
	"sync"
)

func CreateConfigFactory(FileName string, Type string) ConfigInterface {

	yamlConfig := viper.New()
	yamlConfig.AddConfigPath(global.BasePath + "/Config")
	yamlConfig.SetConfigType(Type)
	yamlConfig.SetConfigName(FileName)

	if err := yamlConfig.ReadInConfig(); err != nil {
		panic(global.ErrorsConfigYamlNotExists + err.Error())
	}

	return &ConfigFile{
		Viper: yamlConfig,
		Mu:    new(sync.Mutex),
	}
}
